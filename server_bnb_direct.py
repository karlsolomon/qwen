import json
import os
import time
from contextlib import asynccontextmanager
from typing import List
from uuid import uuid4

import torch
from chromadb import Client as ChromaClient
from chromadb.config import Settings
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Constants ---
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
DEVICE = "cuda:0"
MAX_NEW_TOKENS = 2048
CHUNK_SIZE = 10
HISTORY_FILE = "chat_history.json"
CREATIVE_MODE = True
HISTORY_TOKEN_LIMIT = 50000
HISTORY_TOKEN_TRIM = 25000

# --- Globals ---
model = None
tokenizer = None
chat_history = []
creative_mode = CREATIVE_MODE
chroma_client = ChromaClient(Settings(anonymized_telemetry=False))
doc_collection = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = MAX_NEW_TOKENS
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    stream: bool = False
    creative: bool = CREATIVE_MODE


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, chat_history, doc_collection

    print("🚀 Loading bnb model (direct)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quant_config, trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            chat_history = json.load(f)
    else:
        chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

    doc_collection = chroma_client.get_or_create_collection(name="documents")
    print("📚 Vector store ready.")
    yield


app = FastAPI(lifespan=lifespan)


def count_tokens(history):
    prompt = tokenizer.apply_chat_template(
        history, tokenize=True, add_generation_prompt=False
    )
    return len(prompt)


def trim_chat_history():
    global chat_history
    while count_tokens(chat_history) > HISTORY_TOKEN_LIMIT:
        trimmed = []
        for msg in reversed(chat_history):
            trimmed.insert(0, msg)
            if count_tokens(trimmed) > HISTORY_TOKEN_TRIM:
                trimmed.pop(0)
                break
        chat_history = trimmed


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    global chat_history, creative_mode
    creative_mode = req.creative
    max_tokens = min(req.max_tokens, MAX_NEW_TOKENS)

    chat_history += [m.dict() for m in req.messages]
    trim_chat_history()

    prompt = tokenizer.apply_chat_template(
        chat_history, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    start_time = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )[0]

    generated = output_ids[len(inputs["input_ids"][0]) :]
    reply = tokenizer.decode(generated, skip_special_tokens=True).strip()

    elapsed = time.time() - start_time
    token_count = len(generated)
    print(
        f"⚡ Generated {token_count} tokens in {elapsed:.2f}s ({token_count / elapsed:.2f} tokens/sec)"
    )

    chat_history.append({"role": "assistant", "content": reply})
    with open(HISTORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=2)

    if req.stream:

        def token_generator():
            buffer = ""
            for char in reply:
                buffer += char
                if len(buffer) >= CHUNK_SIZE:
                    yield json.dumps(
                        {
                            "id": "chatcmpl-local-qwen",
                            "object": "chat.completion.chunk",
                            "model": req.model,
                            "choices": [
                                {
                                    "delta": {"role": "assistant", "content": buffer},
                                    "index": 0,
                                }
                            ],
                        }
                    ) + "\n"
                    buffer = ""
            if buffer:
                yield json.dumps(
                    {
                        "id": "chatcmpl-local-qwen",
                        "object": "chat.completion.chunk",
                        "model": req.model,
                        "choices": [
                            {
                                "delta": {"role": "assistant", "content": buffer},
                                "index": 0,
                            }
                        ],
                    }
                ) + "\n"
            yield json.dumps(
                {
                    "id": "chatcmpl-local-qwen",
                    "object": "chat.completion.chunk",
                    "model": req.model,
                    "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                }
            ) + "\n"

        return StreamingResponse(token_generator(), media_type="text/event-stream")

    return {
        "id": "chatcmpl-local-qwen",
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global chat_history
    text = (await file.read()).decode("utf-8", errors="ignore")
    chunks = [text[i : i + 2048] for i in range(0, len(text), 2048)]
    for i, chunk in enumerate(chunks):
        chat_history.append({"role": "user", "content": f"File chunk {i+1}:{chunk}"})
    return {"message": f"Uploaded and chunked {len(chunks)} parts."}


@app.post("/clear")
async def clear_chat():
    global chat_history
    chat_history = [{"role": "system", "content": "You are a helpful assistant."}]
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return {"message": "Chat history cleared."}


@app.post("/creative/{mode}")
async def set_creative_mode(mode: str):
    global creative_mode
    creative_mode = mode.lower() == "on"
    return {"creative_mode": creative_mode}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_bnb_direct:app", host="0.0.0.0", port=11434, reload=False)

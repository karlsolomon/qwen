import json
import os
import time
from contextlib import asynccontextmanager
from typing import List
from uuid import uuid4

from chromadb import Client as ChromaClient
from chromadb.config import Settings
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from gptqmodel import GPTQModel
from pydantic import BaseModel
from transformers import AutoTokenizer

# --- Config ---
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
DEVICE = "cuda:0"
MAX_NEW_TOKENS = 8192
CHUNK_SIZE = 5
HISTORY_FILE = "chat_history.json"
CREATIVE_MODE = True
HISTORY_TOKEN_LIMIT = 50000
HISTORY_TOKEN_TRIM = 25000

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
    print("🚀 Loading model...")
    model = GPTQModel.load(MODEL_NAME, device=DEVICE)
    tokenizer = model.tokenizer
    print("✅ Model loaded.")

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
    max_tokens = min(req.max_tokens, MAX_NEW_TOKENS)
    creative_mode = req.creative
    chat_history += [m.dict() for m in req.messages]
    trim_chat_history()

    prompt = tokenizer.apply_chat_template(
        chat_history, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = (
        tokenizer(prompt, return_tensors="pt").input_ids[0].to(model.model.device)
    )

    start_time = time.time()
    generated_ids = model.generate(prompt, max_new_tokens=max_tokens)[0]
    reply_ids = generated_ids[len(prompt_ids) :]
    reply_text = tokenizer.decode(reply_ids, skip_special_tokens=True)
    elapsed = time.time() - start_time
    token_count = len(reply_ids)
    print(
        f"⚡ Generated {token_count} tokens in {elapsed:.2f}s ({token_count / elapsed:.2f} tokens/sec)"
    )

    chat_history.append({"role": "assistant", "content": reply_text.strip()})
    with open(HISTORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=2)

    if req.stream:

        def token_generator():
            buffer = ""
            for char in reply_text:
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
                                    "finish_reason": None,
                                }
                            ],
                        }
                    ) + "\n"
                    buffer = ""
                    time.sleep(0.01)
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
                                "finish_reason": None,
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
                "message": {"role": "assistant", "content": reply_text.strip()},
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

    uvicorn.run("server:app", host="0.0.0.0", port=11434, reload=False)

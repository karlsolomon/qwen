import asyncio
import json
import os
import threading

import fitz  # PyMuPDF
import torch
import uvicorn
from exllamav2 import ExLlamaV2, ExLlamaV2Cache_Q8, ExLlamaV2Tokenizer
from exllamav2.config import ExLlamaV2Config
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import StreamingResponse

# --- Configs/Flags ---
MAX_PROMPT_TOKENS = 2048
MAX_RESPONSE_TOKENS = 8192
MAX_MEMORY_TOKENS = 32768
context_history = []
current_context_tokens = 0


# --- Globals ---
model = None
tokenizer = None
cache = None
generator = None

device = torch.device("cuda:0")
model_dir = "/home/ksolomon/git/quant"  # <- Update as needed
load_lock = threading.Lock()

# --- FastAPI app ---
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    lazy_load_model()


def format_prompt(user_input: str) -> str:
    return f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"


def lazy_load_model():
    global model, tokenizer, cache, generator

    if model is not None:
        return

    with load_lock:
        if model is not None:
            return  # Already loaded while waiting

        print("🔁 Loading model...")

        # Load config
        config = ExLlamaV2Config()
        config.model_dir = model_dir
        config.prepare()

        # Load model
        loaded_model = ExLlamaV2(config)
        loaded_model.load()

        # Tokenizer, cache, generator
        loaded_tokenizer = ExLlamaV2Tokenizer(config)
        loaded_cache = ExLlamaV2Cache_Q8(loaded_model, lazy=not loaded_model.loaded)
        loaded_generator = ExLlamaV2StreamingGenerator(
            loaded_model, loaded_cache, loaded_tokenizer
        )

        # Assign to globals
        model = loaded_model
        tokenizer = loaded_tokenizer
        cache = loaded_cache
        generator = loaded_generator

        print("✅ Model fully loaded.")


def extract_text_from_pdf(file_path: str) -> str:
    """Extracts all text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"[PDF Error] Failed to extract text: {e}")
        return ""


def chunk_text(text, max_tokens=MAX_PROMPT_TOKENS):
    """Chunk the input text into chunks of max_tokens."""
    chunks = []
    tokens = tokenizer.encode(text)

    while len(tokens) > max_tokens:
        chunk = tokens[:max_tokens]
        tokens = tokens[max_tokens:]
        chunks.append(tokenizer.decode(chunk))

    if tokens.numel() > 0:
        chunks.append(tokenizer.decode(tokens))

    return chunks


@app.post("/upload")
async def upload_file(file: UploadFile):
    if file.filename.endswith(".pdf"):
        contents = await file.read()
        with open("temp.pdf", "wb") as f:
            f.write(contents)
        extracted_text = extract_text_from_pdf("temp.pdf")
        os.remove("temp.pdf")

        if extracted_text:
            chunks = chunk_text(extracted_text)
            for chunk in chunks:
                # Add each chunk to context or process individually
                encoded = tokenizer.encode(chunk)
                add_to_context(len(encoded))
            return {"status": "ok", "chunks": len(chunks)}
        else:
            return {"status": "error", "message": "No text extracted"}

    return {"status": "error", "message": "Unsupported file type"}


def add_to_context(new_chunk_tokens):
    global current_context_tokens
    if current_context_tokens + new_chunk_tokens > MAX_MEMORY_TOKENS:
        while current_context_tokens + new_chunk_tokens > MAX_MEMORY_TOKENS:
            oldest_chunk = context_history.pop(0)
            current_context_tokens -= len(oldest_chunk)

    context_history.append(new_chunk_tokens)
    current_context_tokens += new_chunk_tokens


@app.post("/chat")
async def chat(request: Request):
    lazy_load_model()

    body = await request.json()
    prompt = format_prompt(body.get("prompt", ""))

    # ---- Chunking the prompt if it's more than 2048 tokens ----
    prompt_chunks = chunk_text(prompt)

    # ---- Initialize the model settings ----
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.7
    settings.top_k = 50
    settings.top_p = 0.9
    settings.token_repetition_penalty = 1.1
    settings.eos_token_id = int(tokenizer.eos_token_id or 151643)

    # ---- Process each chunk of prompt separately ----
    for chunk in prompt_chunks:
        encoded = tokenizer.encode(chunk)
        encoded = torch.as_tensor(encoded, dtype=torch.long)

        if encoded.ndim == 1:
            encoded = encoded.unsqueeze(0)
        elif encoded.ndim == 3:
            encoded = encoded.view(encoded.size(0), -1)

        embedding_weight_device = generator.model.modules[0].embedding.weight.device
        input_ids = encoded.contiguous().to(embedding_weight_device)

        print("✅ input_ids.device:", input_ids.device)

        # ---- Add to context ----
        add_to_context(len(encoded))

        # ---- Start streaming generation ----
        generator.begin_stream_ex(input_ids=input_ids, gen_settings=settings)

        async def token_stream():
            while True:
                result = generator.stream_ex()
                text = result.get("chunk", "")
                eos = result.get("eos", False)

                if text:
                    yield json.dumps({"text": text}) + "\n"
                    await asyncio.sleep(0)  # Ensures token flush

                if eos:
                    break

        return StreamingResponse(token_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    # lazy_load_model()
    uvicorn.run("server_exllama:app", host="0.0.0.0", port=8000)

import asyncio
import json
import threading

import torch
import uvicorn
from exllamav2 import ExLlamaV2, ExLlamaV2Cache_Q8, ExLlamaV2Tokenizer
from exllamav2.config import ExLlamaV2Config
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

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
        # model.length = 8192
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


@app.post("/chat")
async def chat(request: Request):
    lazy_load_model()

    body = await request.json()
    prompt = format_prompt(body.get("prompt", ""))

    # ---- Safe tensor creation ----
    encoded = tokenizer.encode(prompt)

    # Convert to tensor if needed
    if not isinstance(encoded, torch.Tensor):
        encoded = torch.tensor(encoded, dtype=torch.long)

    # Guarantee shape [1, seq_len]
    if encoded.ndim == 1:
        encoded = encoded.unsqueeze(0)
    elif encoded.ndim == 3:
        encoded = encoded.view(encoded.size(0), -1)

    # 🔥 Critical: match model's real embedding device
    embedding_weight_device = generator.model.modules[0].embedding.weight.device
    input_ids = encoded.contiguous().to(embedding_weight_device)

    print("✅ FINAL input_ids.device:", input_ids.device)

    # ---- Generation settings ----
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.7
    settings.top_k = 50
    settings.top_p = 0.9
    settings.token_repetition_penalty = 1.1
    settings.eos_token_id = int(tokenizer.eos_token_id or 151643)

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


# @app.post("/chat")
# async def chat(request: Request):
#     lazy_load_model()
#
#     body = await request.json()
#     prompt = format_prompt(body.get("prompt", ""))
#
#     # ---- Safe tensor creation ----
#     encoded = tokenizer.encode(prompt)
#     encoded = torch.as_tensor(encoded, dtype=torch.long)
#
#     if encoded.ndim == 1:
#         encoded = encoded.unsqueeze(0)
#     elif encoded.ndim == 3:
#         encoded = encoded.view(encoded.size(0), -1)
#
#     input_ids = encoded.to(device)
#
#     # ---- Generation settings ----
#     settings = ExLlamaV2Sampler.Settings()
#     settings.temperature = 0.7
#     settings.top_k = 50
#     settings.top_p = 0.9
#     settings.token_repetition_penalty = 1.1
#     settings.eos_token_id = int(tokenizer.eos_token_id or 151643)
#
#     # ---- Start streaming generation ----
#     print("✅ input_ids.shape:", input_ids.shape)
#     print("✅ input_ids.device:", input_ids.device)
#     print("✅ cache.device:", cache.key_states[0].device)
#     print("✅ eos_token_id:", settings.eos_token_id, type(settings.eos_token_id))
#     generator.begin_stream_ex(input_ids=input_ids, gen_settings=settings)
#     print("🧠 sequence_ids.shape:", generator.sequence_ids.shape)
#     print("🧠 sequence_ids.device:", generator.sequence_ids.device)
#     print("🧠 sequence_ids.dtype:", generator.sequence_ids.dtype)
#
#     async def token_stream():
#         while True:
#             result = generator.stream_ex()
#             text = result.get("chunk", "")
#             eos = result.get("eos", False)
#
#             if text:
#                 yield json.dumps({"text": text}) + "\n"
#                 await asyncio.sleep(0)  # Ensures token flush
#
#             if eos:
#                 break
#
#     return StreamingResponse(token_stream(), media_type="text/event-stream")
#

if __name__ == "__main__":
    # lazy_load_model()
    uvicorn.run("server_exllama:app", host="0.0.0.0", port=8000)

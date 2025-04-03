
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import torch
import time
import json

from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2SamplerSettings
from exllamav2.tokenizer import ExLlamaV2Tokenizer
from exllamav2.cache import ExLlamaV2Cache_Q8
from model_init import init as model_init  # your model_init.py should have this

app = FastAPI()

model = None
tokenizer = None
generator = None

class DummyArgs:
    max_new_tokens = 512
    temperature = 0.7
    top_p = 0.95
    top_k = 0
    repetition_penalty = 1.1
    token_repetition_penalty = 1.1
    mirostat = 0
    mirostat_tau = 5.0
    mirostat_eta = 0.1
    tfs = 1.0
    typical = 1.0
    stream = True
    stop = []

args = DummyArgs()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, generator

    print("🚀 Loading ExLlamaV2 model...")
    model, tokenizer = model_init(args)
    cache = ExLlamaV2Cache_Q8(model, lazy=not model.loaded)
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
    print(f"[DEBUG] model.device_context: {model.device_context}")
    yield

app.router.lifespan_context = lifespan

@app.post("/v1/chat/completions")
async def chat(request: Request):
    global model, tokenizer, generator

    body = await request.json()
    messages = body.get("messages", [])
    prompt_text = messages[-1]["content"] if messages else ""

    input_ids = tokenizer.encode(prompt_text)
    dev = model.device_context[0]
    target_device = torch.device(f"cuda:{dev.device_idx}")
    input_ids = torch.tensor([input_ids]).to(target_device)

    generator.warmup()
    generator.set_stop_conditions([])
    
    settings = ExLlamaV2SamplerSettings()
    settings.temperature = args.temperature
    settings.top_k = args.top_k
    settings.top_p = args.top_p
    settings.token_repetition_penalty = args.token_repetition_penalty
    settings.disallow_tokens = []
    settings.seed = -1
    settings.typical = args.typical
    settings.tfs = args.tfs
    settings.mirostat = args.mirostat
    settings.mirostat_tau = args.mirostat_tau
    settings.mirostat_eta = args.mirostat_eta
    settings.penalty_repeat = args.repetition_penalty

    generator.begin_stream(input_ids, settings)

    async def stream_response():
        start = time.time()
        total_tokens = 0
        first = True
        try:
            while True:
                chunk, eos, _, _, _, _, _, _ = generator.stream()
                if chunk:
                    text = tokenizer.decode(chunk)
                    total_tokens += len(chunk)
                    data = {
                        "id": "chatcmpl-xyz",
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "delta": {"content": text},
                            "index": 0,
                            "finish_reason": None if not eos else "stop"
                        }],
                    }
                    if first:
                        yield "data: " + json.dumps(data) + "\n\n"
                        first = False
                    else:
                        yield "data: " + json.dumps(data) + "\n\n"
                if eos:
                    break
            yield "data: [DONE]\n\n"
        finally:
            end = time.time()
            print(f"⚡ Generated {total_tokens} tokens in {end - start:.2f}s ({(total_tokens / (end - start)) if end > start else 0:.2f} tokens/sec)")

    return StreamingResponse(stream_response(), media_type="text/event-stream")

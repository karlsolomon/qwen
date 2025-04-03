import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
from exllamav2 import ExLlamaV2Cache_Q8, ExLlamaV2Tokenizer, model_init
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


class DummyArgs:
    model_dir = "/home/ksolomon/git/quant/"
    gpu_split = None
    tensor_parallel = False
    length = 4096
    rope_scale = None
    rope_alpha = None
    rope_yarn = None
    no_flash_attn = False
    no_xformers = False
    no_sdpa = False
    no_graphs = False
    low_mem = True
    experts_per_token = None
    load_q4 = False
    fast_safetensors = False
    ignore_compatibility = True
    chunk_size = None
    draft_model_dir = None
    no_draft_scale = False
    draft_n_tokens = 5
    cache_8bit = False
    cache_q4 = False
    cache_q6 = False
    cache_q8 = True
    temperature = 0.8
    top_k = 50
    top_p = 0.95
    top_a = 0.0
    typical = 0.0
    repetition_penalty = 1.1
    frequency_penalty = 0.0
    presence_penalty = 0.0
    smoothing_factor = 0.0
    xtc_probability = 0.0
    xtc_threshold = 0.1
    dry_allowed_length = 2
    dry_base = 1.75
    dry_multiplier = 0.0
    dry_range = 0
    dynamic_temperature = None
    mode = "chatml"
    username = "User"
    botname = "Bot"
    system_prompt = None
    no_system_prompt = False
    modes = False
    max_response_tokens = 1000
    response_chunk = 5
    print_timings = True
    amnesia = False
    ngram_decoding = False


args = DummyArgs()

app = FastAPI()
model, tokenizer, generator = None, None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, generator

    print("🚀 Loading ExLlamaV2 model...")
    model, tokenizer = model_init.init(args)
    cache = ExLlamaV2Cache_Q8(model, lazy=not model.loaded)
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
    yield


app.router.lifespan_context = lifespan


@app.post("/v1/chat/completions")
async def chat(request: Request):
    global generator, tokenizer

    body = await request.json()
    messages = body["messages"]
    prompt_text = messages[-1]["content"]

    input_ids = tokenizer.encode(prompt_text, add_bos=True)
    target_device = torch.device(f"cuda:{model.device_context[0].device_idx}")
    input_ids = input_ids.to(target_device)

    settings = ExLlamaV2Sampler.Settings(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        top_a=args.top_a,
        typical=args.typical,
        token_repetition_penalty=args.repetition_penalty,
        token_frequency_penalty=args.frequency_penalty,
        token_presence_penalty=args.presence_penalty,
        smoothing_factor=args.smoothing_factor,
        xtc_probability=args.xtc_probability,
        xtc_threshold=args.xtc_threshold,
        dry_allowed_length=args.dry_allowed_length,
        dry_base=args.dry_base,
        dry_multiplier=args.dry_multiplier,
        dry_range=args.dry_range,
    )

    if args.dynamic_temperature:
        dt_args = [float(v) for v in args.dynamic_temperature.split(",")]
        settings.min_temp = dt_args[0]
        settings.max_temp = dt_args[1]
        settings.temp_exponent = dt_args[2]

    embedding_weight_device = generator.model.modules[0].embedding.weight.device
    input_ids = input_ids.to(embedding_weight_device)
    generator.begin_stream_ex(input_ids, settings)

    async def stream_response() -> AsyncGenerator[str, None]:
        response_text = ""
        response_tokens = 0
        time_start = time.time()

        while True:
            res = generator.stream_ex()
            chunk = res["chunk"]
            eos = res["eos"]
            tokens = res["chunk_token_ids"]

            if len(response_text) == 0:
                chunk = chunk.lstrip()
            response_text += chunk

            yield chunk
            response_tokens += len(tokens[0])

            if eos or response_tokens >= args.max_response_tokens:
                break

        if args.print_timings:
            time_end = time.time()
            elapsed = time_end - time_start
            print(
                f"⚡ Generated {response_tokens} tokens in {elapsed:.2f}s ({response_tokens/elapsed:.2f} tokens/sec)"
            )

    return StreamingResponse(stream_response(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server_exllama:app",
        host="0.0.0.0",
        port=11434,
        reload=False,
    )

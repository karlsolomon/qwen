# server.py: FastAPI + ExLlamaV2 LLM server with streaming and special command handling

import asyncio
import json
import os
import time
from typing import Generator

import safetensors
import torch
import uvicorn
from exllamav2 import ExLlamaV2, ExLlamaV2Cache_Q8, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2Sampler
from exllamav2.generator.streaming import ExLlamaV2StreamingGenerator
from exllamav2.model_init import init as model_init
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from safetensors.torch import load_file, save_file

# Configuration constants
INSTRUCTION_LIMIT = 4096
CHAT_CONTEXT_LIMIT = 28672
PROMPT_LIMIT = 2048
RESPONSE_LIMIT = 4096
MODEL_DIR = "/home/ksolomon/git/quant"
SAFETENSORS_FILE = MODEL_DIR + "/model.safetensors"
CUDA_DEVICE = "cuda:0"
CHUNK_SIZE = 5

# Global context buffers
instruction_context = []
chat_context = []

# Global Exllama variables
model = None
tokenizer = None
generator = None
settings = None
cache = None


# Load instruction memory from disk
INSTRUCTION_FILE = "instruction_context.txt"
if os.path.exists(INSTRUCTION_FILE):
    with open(INSTRUCTION_FILE, "r") as f:
        instruction_context = f.readlines()

# FastAPI app setup
app = FastAPI()


def get_model_params(model):
    # Assume `model` is your trained neural network and it's already loaded onto GPU `cuda:0`
    # Access the parameters as a Python generator
    params = [param for param in model.parameters()]

    # If you want to inspect or modify one specific parameter:
    for name, param in model.named_parameters():
        print(f"Parameter {name} is located at {param.device}")

    # You might want to copy all parameters to CPU before further processing
    params_cpu = [p.cpu() for p in params]

    # Example of saving model state_dict
    state_dict = model.state_dict()
    torch.save(state_dict, "model_state.pth")

    # Alternatively, if you just want a dictionary containing all tensors:
    params_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    for k, v in params_dict:
        print(f"{k}:{v}")


@app.post("/save")
def save_checkpoint(model, filename="checkpoint.safetensors"):
    """Save model state_dict using safetensors."""
    state_dict = model.state_dict()
    save_file(state_dict, filename)
    print(f"Checkpoint saved at epoch: {filename}")


@app.post("/load")
def load_checkpoint(filename="checkpoint.safetensors", model=None):
    """Load model state_dict from safetensors file."""
    state_dict = load_file(filename)
    assert isinstance(model, torch.nn.Module), "Please provide a valid torch model."
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from: {filename}")


# Input model for chat
class ChatRequest(BaseModel):
    prompt: str


def lazy_load_model():
    # Load model + tokenizer
    print("🔁 Loading model...")
    args = type(
        "Args",
        (),
        {
            "model_dir": MODEL_DIR,
            "gpu_split": None,
            "tensor_parallel": False,
            "length": 4096,
            "rope_scale": None,
            "rope_alpha": None,
            "rope_yarn": None,
            "no_flash_attn": False,
            "no_xformers": False,
            "no_sdpa": False,
            "no_graphs": False,
            "low_mem": False,
            "experts_per_token": None,
            "load_q4": False,
            "load_q8": True,
            "fast_safetensors": False,
            "ignore_compatibility": True,
            "chunk_size": PROMPT_LIMIT,
        },
    )()
    global model, tokenizer, generator, settings, cache
    model, tokenizer = model_init(
        args, progress=True, max_input_len=PROMPT_LIMIT, max_output_len=RESPONSE_LIMIT
    )

    settings = ExLlamaV2Sampler().Settings()
    settings.temperature = 0.8
    settings.top_k = 50
    settings.top_p = 0.95
    settings.token_repetition_penalty = 1.1
    settings.eos_token_id = tokenizer.eos_token_id or 151643

    cache = ExLlamaV2Cache_Q8(model, lazy=not model.loaded)

    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
    print("✅ Model fully loaded.")


# Util to manage instruction token space
def trim_instruction_context():
    while tokenizer.num_tokens("\n".join(instruction_context)) > INSTRUCTION_LIMIT:
        instruction_context.pop(0)
    with open(INSTRUCTION_FILE, "w") as f:
        f.writelines(instruction_context)


def normalize_decoded(output):
    return "".join(output) if isinstance(output, list) else output


# Core streaming logic
def generate_stream(prompt: str) -> Generator[str, None, None]:
    global generator, tokenizer, model, settings
    start = time.time()
    tokens = 0
    buffer = []

    input_tokens = tokenizer.encode(prompt, add_bos=True, add_eos=True)
    generator.begin_stream_ex(input_tokens, settings)

    while True:
        result = generator.stream_ex()
        chunk_ids = result.get("chunk_token_ids", None)
        eos = result.get("eos", False)

        if chunk_ids is not None and chunk_ids.numel() > 0:
            for token in chunk_ids[0].tolist():
                buffer.append(token)
                tokens += 1
                if len(buffer) >= CHUNK_SIZE:
                    decoded = tokenizer.decode(torch.tensor(buffer).unsqueeze(0))
                    yield f"data:{json.dumps({'text': decoded})}\n\n"
                    buffer.clear()

        if eos:
            break

    if buffer:
        final_decoded = tokenizer.decode(torch.tensor(buffer).unsqueeze(0))
        yield f"data:{json.dumps({'text': final_decoded})}\n\n"

    yield f"data:{json.dumps({'text': '[DONE]'})}\n\n"

    tps = tokens / (time.time() - start)
    print(f"[Server] {tokens} tokens streamed at {tps:.2f} tokens/sec")


@app.post("/chat")
async def chat(request: ChatRequest):
    prompt = request.prompt.strip()

    if prompt.startswith("/clear"):
        chat_context.clear()
        return JSONResponse(content={"message": "Chat context cleared."})

    elif prompt.startswith("/clearall"):
        chat_context.clear()
        instruction_context.clear()
        open(INSTRUCTION_FILE, "w").close()
        return JSONResponse(content={"message": "All context cleared."})

    elif prompt.startswith("/instruct"):
        instruction = prompt[len("/instruct") :].strip()
        instruction_context.append(instruction)
        trim_instruction_context()
        return JSONResponse(content={"message": "Instruction added."})

    elif prompt.startswith("/getfiletypes"):
        return JSONResponse(content=".txt,.pdf,.zip,.tar.gz")

    elif prompt.startswith("/load"):
        load_checkpoint(model, SAFETENSORS_FILE)
        return JSONResponse(
            content={"message": "Load .safetensors context from previous chat."}
        )

    elif prompt.startswith("/save"):
        save_checkpoint(model, SAFETENSORS_FILE)
        return JSONResponse(
            content={
                "message": "Save .safetensors context to be loaded in future chats."
            }
        )

    elif prompt.startswith("/upload"):
        return JSONResponse(
            content={"message": "Upload logic not yet implemented in this stub."}
        )

    chat_context.append(prompt)
    return JSONResponse(content={"message": "Prompt received."})


@app.post("/stream")
async def stream_chat(request: ChatRequest):
    return StreamingResponse(
        generate_stream(request.prompt), media_type="text/event-stream"
    )


@app.on_event("startup")
async def startup_event():
    lazy_load_model()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

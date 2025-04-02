import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

# --- Constants ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"
DEVICE = "cuda:0"
MAX_NEW_TOKENS = 2048

# --- Load tokenizer and model with bitsandbytes 4-bit quant ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": 0},
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# --- FastAPI setup ---
app = FastAPI()


# --- OpenAI-style request/response ---
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = MAX_NEW_TOKENS
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    stream: bool = False


@app.post("/v1/chat/completions")
async def chat_endpoint(req: ChatRequest):
    # Apply template
    prompt = tokenizer.apply_chat_template(
        [m.dict() for m in req.messages], tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(
        tokenizer, skip_special_tokens=True, skip_prompt=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=req.max_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Generate in background
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Collect streamed output
    output_text = ""
    for token in streamer:
        output_text += token

    # OpenAI-compatible response
    return {
        "id": "chatcmpl-local-qwen",
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text.strip(),
                },
                "finish_reason": "stop",
            }
        ],
    }


# --- Run ---
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=11434, reload=False)

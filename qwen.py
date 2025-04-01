# qwen4bit.py

import json
import os
import subprocess
import threading
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

# Constants
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ"
DEVICE = "cuda:0"
CREATIVE = True  # Toggle between sampling and greedy
MAX_NEW_TOKENS = 2048
MAX_TOTAL_TOKENS = 16384
FILE_CHUNK_SIZE = 2048
HISTORY_PATH = "chat_history.json"

# BitsAndBytes Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


def count_tokens(history, tokenizer):
    tokenized = tokenizer.apply_chat_template(
        history, tokenize=True, add_generation_prompt=False
    )
    return len(tokenized)


def truncate_history(chat_history, tokenizer, max_tokens=MAX_TOTAL_TOKENS // 2):
    new_history = []
    for i in range(len(chat_history) - 1, -1, -1):
        new_history.insert(0, chat_history[i])
        if count_tokens(new_history, tokenizer) > max_tokens:
            new_history.pop(0)  # remove oldest until within limit
            break
    return new_history


# Load Tokenizer and Model


def load_model_and_tokenizer(model_name, device, bnb_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": int(DEVICE[len(DEVICE) - 1])},
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    return tokenizer, model


# Load Chat History
def load_chat_history(history_path):
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            chat_history = json.load(f)
        print("📂 Loaded previous chat history.")
    else:
        chat_history = [
            {"role": "system", "content": "You are a helpful coding assistant."}
        ]
    return chat_history


# Save Chat History
def save_chat_history(chat_history, history_path):
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)


# Open File with Dolphin
def open_file_with_thunar():
    print("📁 Please select a file in Dolphin... (then close the window to continue)")

    # Launch Dolphin and block until user closes it
    thunar_proc = subprocess.Popen(
        ["thunar", "--select", ".", ">/dev/null", "2>&1"], stdout=subprocess.PIPE
    )
    thunar_proc.wait()

    # Ask user to paste path of selected file
    filepath = input("📎 Enter full path to the file you selected: ").strip()
    if not os.path.exists(filepath):
        print("❌ File not found.")
        return None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# Generate Response
def generate_response(tokenizer, model, prompt, device, creative, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    token_count = inputs["input_ids"].shape[1]
    print(f"🧮 Prompt token count: {token_count}")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    start_time = time.time()

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )

    if creative:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            }
        )
    else:
        generation_kwargs.update({"do_sample": False})

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # --- Stream output
    print("🤖 Qwen: ", end="", flush=True)
    response_tokens = []
    for token in streamer:
        print(token, end="", flush=True)
        response_tokens.append(token)

    # --- Done generating
    print()  # newline after response
    end_time = time.time()

    response_text = "".join(response_tokens).strip()
    gen_token_count = tokenizer(response_text, return_tensors="pt")["input_ids"].shape[
        1
    ]
    elapsed_time = end_time - start_time
    tokens_per_sec = (
        gen_token_count / elapsed_time if elapsed_time > 0 else float("inf")
    )

    print(
        f"\n⚡ Generated {gen_token_count} tokens in {elapsed_time:.2f} sec ({tokens_per_sec:.2f} tokens/sec)"
    )

    return response_text


# Main Function
def main():
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME, DEVICE, bnb_config)
    chat_history = load_chat_history(HISTORY_PATH)

    print("💬 Qwen Chat Interface (type 'exit' to quit)")
    while True:
        user_input = input("🧑 You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break
        if user_input.strip().lower() == "/ul":
            file_content = open_file_with_thunar()
            if file_content:
                # Tokenize the file to check its length
                tokens = tokenizer(file_content, return_tensors="pt")["input_ids"][0]
                token_chunks = torch.split(tokens, FILE_CHUNK_SIZE)

                for i, chunk in enumerate(token_chunks):
                    chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
                    print(
                        f"Chunk uses {len(tokenizer(chunk_text)['input_ids'])} tokens."
                    )
                    chat_history.append(
                        {
                            "role": "user",
                            "content": f"Chunk {i + 1} of uploaded file:\n\n{chunk_text}",
                        }
                    )
                print(f"✅ Uploaded file split into {len(token_chunks)} chunks.")
            continue

        chat_history.append({"role": "user", "content": user_input})

        # Truncate if chat history exceeds MAX_TOTAL_TOKENS
        if count_tokens(chat_history, tokenizer) > MAX_TOTAL_TOKENS:
            chat_history = truncate_history(chat_history, tokenizer)
            print("🧹 Chat history truncated to stay within token limit.")
            print(f"🧮 Total prompt tokens: {count_tokens(chat_history, tokenizer)}")
        prompt = tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

        response_text = generate_response(
            tokenizer, model, prompt, DEVICE, CREATIVE, MAX_NEW_TOKENS
        )

        chat_history.append({"role": "assistant", "content": response_text})
        save_chat_history(chat_history, HISTORY_PATH)


if __name__ == "__main__":
    main()

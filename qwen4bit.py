from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
import torch
import time
import threading
import os
import json
import subprocess


def open_file_with_dolphin():
    print("📁 Please select a file in Dolphin... (then close the window to continue)")

    # Launch Dolphin and block until user closes it
    dolphin_proc = subprocess.Popen(["dolphin", "--select", "."], stdout=subprocess.PIPE)
    dolphin_proc.wait()

    # Ask user to paste path of selected file
    filepath = input("📎 Enter full path to the file you selected: ").strip()
    if not os.path.exists(filepath):
        print("❌ File not found.")
        return None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# --- Model Config ---
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
device = "cuda:0"

CREATIVE = False  # Toggle between sampling and greedy
MAX_NEW_TOKENS = 2048

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ✅ Load tokenizer and model BEFORE using them
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0},
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# ✅ Define chat history BEFORE chat loop
# --- Persistent chat history setup ---
HISTORY_PATH = "chat_history.json"

if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        chat_history = json.load(f)
    print("📂 Loaded previous chat history.")
else:
    chat_history = [{"role": "system", "content": "You are a helpful coding assistant."}]

# ✅ Chat loop starts here
print("💬 Qwen Chat Interface (type 'exit' to quit)")
while True:
    user_input = input("🧑 You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        break
    if user_input.strip().lower() == "/ul":
        file_content = open_file_with_dolphin()
        if file_content:
            chat_history.append({"role": "user", "content": f"Please read and understand the following file:\n\n{file_content}"})
            print("✅ File content added to context.")
        continue

    chat_history.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    token_count = inputs["input_ids"].shape[1]
    print(f"🧮 Prompt token count: {token_count}")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    start_time = time.time()
    
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    if CREATIVE:
        generation_kwargs.update({
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
        })
    else:
        generation_kwargs.update({
            "do_sample": False
        })
    
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
    gen_token_count = tokenizer(response_text, return_tensors="pt")["input_ids"].shape[1]
    elapsed_time = end_time - start_time
    tokens_per_sec = gen_token_count / elapsed_time if elapsed_time > 0 else float("inf")
    
    print(f"\n⚡ Generated {gen_token_count} tokens in {elapsed_time:.2f} sec ({tokens_per_sec:.2f} tokens/sec)")
    
    chat_history.append({"role": "assistant", "content": response_text})
    # Save updated history
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)
	

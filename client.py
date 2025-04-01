# qwen.py — Client that connects to local Qwen FastAPI Server

import json
import os
import subprocess
import time

import requests

# Constants
API_URL = "http://localhost:11434/v1/chat/completions"
MODEL_ID = "qwen-coder-32b"
CREATIVE = True
MAX_NEW_TOKENS = 2048
MAX_TOTAL_TOKENS = 16384
FILE_CHUNK_SIZE = 2048
HISTORY_PATH = "chat_history.json"


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


# Open File with Thunar
def open_file_with_thunar():
    print("📁 Please select a file in Thunar... (then close the window to continue)")
    thunar_proc = subprocess.Popen(["thunar", "--select", "."], stdout=subprocess.PIPE)
    thunar_proc.wait()
    filepath = input("📎 Enter full path to the file you selected: ").strip()
    if not os.path.exists(filepath):
        print("❌ File not found.")
        return None
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# Truncate Chat History if Needed
def count_tokens(history):
    return sum(len(m["content"].split()) for m in history)


def truncate_history(chat_history, max_tokens=MAX_TOTAL_TOKENS // 2):
    new_history = []
    for i in range(len(chat_history) - 1, -1, -1):
        new_history.insert(0, chat_history[i])
        if count_tokens(new_history) > max_tokens:
            new_history.pop(0)
            break
    return new_history


# Main Function
def main():
    chat_history = load_chat_history(HISTORY_PATH)

    print("💬 Qwen Chat Interface (connected to FastAPI, type 'exit' to quit)")
    while True:
        user_input = input("🧑 You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break
        if user_input.strip().lower() == "/ul":
            file_content = open_file_with_thunar()
            if file_content:
                words = file_content.split()
                chunks = [
                    " ".join(words[i : i + FILE_CHUNK_SIZE])
                    for i in range(0, len(words), FILE_CHUNK_SIZE)
                ]
                for i, chunk in enumerate(chunks):
                    chat_history.append(
                        {
                            "role": "user",
                            "content": f"Chunk {i + 1} of uploaded file:\n\n{chunk}",
                        }
                    )
                print(f"✅ Uploaded file split into {len(chunks)} chunks.")
            continue

        chat_history.append({"role": "user", "content": user_input})

        if count_tokens(chat_history) > MAX_TOTAL_TOKENS:
            chat_history = truncate_history(chat_history)
            print("🧹 Chat history truncated to stay within token limit.")

        payload = {
            "model": MODEL_ID,
            "messages": chat_history,
            "temperature": 0.7 if CREATIVE else None,
            "top_p": 0.8 if CREATIVE else None,
            "top_k": 20 if CREATIVE else None,
            "max_tokens": MAX_NEW_TOKENS,
            "stream": False,
        }

        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        end_time = time.time()
        elapsed = end_time - start_time

        if response.ok:
            reply = response.json()["choices"][0]["message"]["content"].strip()
            tokens_generated = len(reply.split())
            tps = tokens_generated / elapsed if elapsed > 0 else float("inf")
            print(f"🤖 Qwen: {reply}")
            print(
                f"⚡ Generated {tokens_generated} tokens in {elapsed:.2f}s ({tps:.2f} tokens/sec)"
            )
            chat_history.append({"role": "assistant", "content": reply})
        else:
            print("❌ API Error:", response.status_code, response.text)

        save_chat_history(chat_history, HISTORY_PATH)


if __name__ == "__main__":
    main()

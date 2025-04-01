import json

import requests

API_URL = "http://localhost:11434/v1/chat/completions"
MODEL_ID = "qwen-coder-32b"

# Initial system prompt
chat_history = [{"role": "system", "content": "You are a helpful coding assistant."}]

print("💬 Connected to Qwen API at http://localhost:11434")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("🧑 You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    chat_history.append({"role": "user", "content": user_input})

    payload = {
        "model": MODEL_ID,
        "messages": chat_history,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 2048,
        "stream": False,
    }

    response = requests.post(API_URL, json=payload)
    data = response.json()

    if "choices" in data:
        reply = data["choices"][0]["message"]["content"].strip()
        print(f"🤖 Qwen: {reply}\n")
        chat_history.append({"role": "assistant", "content": reply})
    else:
        print("❌ Error:", data)

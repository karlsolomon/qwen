import json
import os

import requests


def main():
    print("💬 Qwen Chat Interface (streaming, type 'exit' to quit)\n")
    history = []

    while True:
        user_input = input("🧑 You: ")
        if user_input.strip().lower() == "exit":
            break
        if user_input.startswith("/upload "):
            path = user_input.split("/upload ", 1)[1].strip()
            with open(path, "rb") as f:
                files = {"file": (os.path.basename(path), f)}
                res = requests.post("http://localhost:11434/upload", files=files)
                print(f"📤 {res.json()['message']}")
            continue

        if user_input == "/clear":
            requests.post("http://localhost:11434/clear")
            print("🧹 Chat history cleared.")
            history = []
            continue

        if user_input.startswith("/creative"):
            mode = user_input.split("/creative", 1)[1].strip()
            requests.post(f"http://localhost:11434/creative/{mode}")
            print(f"🎨 Creative mode set to: {mode}")
            continue

        history.append({"role": "user", "content": user_input})

        payload = {
            "model": "Qwen2.5-14B-Instruct-GPTQ",
            "messages": history,
            "stream": True,
        }

        print("🤖 Qwen: ", end="", flush=True)
        response = requests.post(
            "http://localhost:11434/v1/chat/completions", json=payload, stream=True
        )

        reply = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    delta = data["choices"][0]["delta"]
                    if "content" in delta:
                        token = delta["content"]
                        print(token, end="", flush=True)
                        reply += token
                except Exception as e:
                    print(f"⚠️ Error decoding line: {line} -> {e}")

        print()
        history.append({"role": "assistant", "content": reply.strip()})


if __name__ == "__main__":
    main()

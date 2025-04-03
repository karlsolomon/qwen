import json

import requests

API_URL = "http://localhost:11434/v1/chat/completions"


def stream_chat():
    print("💬 Qwen Chat Interface (OpenAI-style streaming, type 'exit' to quit)\n")

    while True:
        user_input = input("🧑 You: ").strip()
        if user_input.lower() == "exit":
            break

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "qwen2.5-14b-exllama",  # adjust name as needed
            "messages": [{"role": "user", "content": user_input}],
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "stream": True,
        }

        print("🤖 Qwen: ", end="", flush=True)
        try:
            response = requests.post(
                API_URL, headers=headers, json=payload, stream=True
            )

            for line in response.iter_lines():
                if line:
                    if line.startswith(b"data: "):
                        try:
                            chunk = json.loads(line[6:].decode("utf-8"))
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta:
                                print(delta["content"], end="", flush=True)
                        except json.JSONDecodeError:
                            print(f"\n⚠️ Error decoding line: {line}")
                    elif line == b"[DONE]":
                        break
        except KeyboardInterrupt:
            print("\n✋ Interrupted by user")
            break

        print()  # newline after model response


if __name__ == "__main__":
    stream_chat()

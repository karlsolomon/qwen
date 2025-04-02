
import requests
import json

def main():
    print("💬 Qwen Chat Interface (bnb streaming, type 'exit' to quit)\n")
    history = []

    while True:
        user_input = input("🧑 You: ")
        if user_input.strip().lower() == "exit":
            break

        history.append({"role": "user", "content": user_input})

        payload = {
            "model": "Qwen2.5-14B-Instruct",
            "messages": history,
            "stream": True
        }

        print("🤖 Qwen: ", end="", flush=True)
        response = requests.post("http://localhost:11434/v1/chat/completions", json=payload, stream=True)

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

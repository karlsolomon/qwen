import asyncio
import json
import os
import time

import httpx
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

console = Console()
history_path = "chat_history.json"
chat_history = []
file_to_append = None


def load_chat_history():
    global chat_history
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                chat_history = json.load(f)
            console.print(
                f"💾 Loaded [cyan]{len(chat_history)}[/cyan] past turns from history."
            )
        except Exception as e:
            console.print(f"[red]Failed to load history:[/red] {e}")


def save_chat_history():
    try:
        with open(history_path, "w") as f:
            json.dump(chat_history, f, indent=2)
    except Exception as e:
        console.print(f"[red]Failed to save chat:[/red] {e}")


def clear_chat_history():
    global chat_history
    chat_history = []
    if os.path.exists(history_path):
        os.remove(history_path)
    console.print("[yellow]⚠ Chat history cleared.[/yellow]")


def build_prompt():
    prompt = ""
    for turn in chat_history:
        prompt += f"<|im_start|>user\n{turn['user']}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{turn['assistant']}<|im_end|>\n"
    return prompt


def handle_file_upload() -> str:
    file_path = console.input("[bold yellow]📄 Enter file path: [/bold yellow]").strip()
    if not os.path.isfile(file_path):
        console.print(f"[red]File not found:[/red] {file_path}")
        return ""

    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        try:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "application/pdf")}
                response = httpx.post("http://localhost:8000/upload", files=files)
                resp = response.json()
                if resp.get("status") == "ok":
                    console.print(f"[green]✔ Uploaded PDF:[/green] {file_path}")
                else:
                    console.print(f"[red]❌ Server error:[/red] {resp.get('message')}")
        except Exception as e:
            console.print(f"[red]Upload failed:[/red] {e}")
        return ""  # PDFs go to server context, nothing added to user prompt

    else:
        try:
            with open(file_path, "r") as f:
                content = f.read()
            console.print(f"[green]✔ Loaded file:[/green] {file_path}")
            return f"\n--- FILE: {os.path.basename(file_path)} ---\n{content}\n--- END FILE ---\n"
        except Exception as e:
            console.print(f"[red]Error reading file:[/red] {e}")
            return ""


async def stream_chat_async(client, full_prompt: str) -> str:
    url = "http://localhost:8000/chat"
    headers = {"Content-Type": "application/json"}
    data = {"prompt": full_prompt}

    token_count = 0
    response_text = ""
    rendered_text = ""
    start_time = time.time()

    async with client.stream(
        "POST", url, headers=headers, json=data, timeout=None
    ) as response:
        with Live(Markdown(""), refresh_per_second=15, console=console) as live:
            buffer = ""
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    text = payload.get("text", "")
                    response_text += text
                    rendered_text += text
                    token_count += 1
                    live.update(Markdown(rendered_text))
                except json.JSONDecodeError:
                    continue

    elapsed = time.time() - start_time
    if token_count > 0:
        console.print(
            f"\n⚡ {token_count} tokens in {elapsed:.2f}s — {token_count / elapsed:.2f} tokens/sec\n",
            style="dim",
        )

    return response_text


async def chat_loop():
    global file_to_append
    load_chat_history()
    console.print("💬 [bold]Chat with Qwen 2.5 (Ctrl+C to quit)[/bold]")
    console.print(
        "✨ Type [cyan]/upload[/cyan] to attach a file, [red]/clear[/red] to reset session.\n"
    )

    async with httpx.AsyncClient() as client:
        while True:
            try:
                user_input = await asyncio.to_thread(
                    console.input, "\n[bold blue]🧑‍💻[/bold blue]: "
                )
                user_input = user_input.strip()

                if user_input == "/upload":
                    file_to_append = handle_file_upload()
                    continue

                elif user_input == "/clear":
                    clear_chat_history()
                    file_to_append = None
                    continue

                prompt_text = user_input
                if file_to_append:
                    prompt_text += "\n" + file_to_append
                    file_to_append = None

                full_prompt = (
                    build_prompt()
                    + f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
                )

                console.print("[bold magenta]🤖[/bold magenta]:")
                response = await stream_chat_async(client, full_prompt)

                chat_history.append(
                    {"user": prompt_text.strip(), "assistant": response.strip()}
                )
                save_chat_history()

            except KeyboardInterrupt:
                console.print("\n👋 [bold red]Goodbye![/bold red]")
                break
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}")


if __name__ == "__main__":
    asyncio.run(chat_loop())

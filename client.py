# client.py: Interactive rich client for chatting with the FastAPI LLM server

import argparse
import os

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text

console = Console()

API_URL = "http://localhost:8000"


def get_supported_filetypes():
    response = httpx.post(f"{API_URL}/chat", json={"prompt": "/getfiletypes"})
    return response.text.strip().split(",")


def upload_file(path):
    if not os.path.isfile(path):
        console.print(f"[red]File not found:[/red] {path}")
        return
    filetypes = get_supported_filetypes()
    ext = os.path.splitext(path)[1]
    if ext not in filetypes:
        console.print(f"[red]Unsupported file type:[/red] {ext}")
        return
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    response = httpx.post(f"{API_URL}/chat", json={"prompt": f"/upload {path}"})
    if response.status_code == 200:
        console.print(f"[green]Uploaded:[/green] {path}")
    else:
        console.print(f"[red]Upload failed:[/red] {response.text}")


def stream_chat(prompt):
    with httpx.stream(
        "POST",
        f"{API_URL}/stream",
        json={"prompt": prompt},
        headers={"Accept": "text/event-stream"},
        timeout=None,
    ) as response:
        if response.status_code != 200:
            console.print(f"[red]Error:[/red] {response.text}")
            return
        console.print("[blue]Assistant:[/blue]", end=" ", soft_wrap=True)
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("data:"):
                try:
                    decoded = line[len("data:") :].strip()
                    console.print(decoded, end="", soft_wrap=True)
                except Exception as e:
                    console.print(f"[red]Decode error:[/red] {e}")
        console.print("")


def main():
    console.print("[bold green]Local LLM Client[/bold green] 🧠")
    console.print(
        "Type [bold]/exit[/bold] to quit. Use [bold]/clear[/bold], [bold]/upload <path>[/bold], etc."
    )
    while True:
        user_input = Prompt.ask("[yellow]You[/yellow]")
        if user_input.strip() == "/exit":
            break
        elif user_input.startswith("/upload"):
            _, path = user_input.split(maxsplit=1)
            upload_file(path.strip())
        else:
            stream_chat(user_input)


if __name__ == "__main__":
    main()

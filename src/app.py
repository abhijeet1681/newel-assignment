from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .rag.rag_pipeline import answer_question


console = Console()


def main() -> None:
    console.print(Panel.fit("Swiggy Annual Report RAG QA (CLI)", style="bold"))
    console.print("Type a question. Type 'exit' to quit.\n")

    while True:
        q = Prompt.ask("[bold cyan]Question[/bold cyan]")
        if q.strip().lower() in {"exit", "quit"}:
            break

        try:
            answer, chunks = answer_question(q, k=4)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            console.print("Did you run: python scripts/build_index.py --pdf data/Swiggy_Annual_Report.pdf ?\n")
            continue

        console.print(Panel(answer, title="Answer", style="green"))
        console.print("[bold]Top supporting chunks:[/bold]")
        for i, c in enumerate(chunks, start=1):
            console.print(f"{i}. (Page {c.page}, score {c.score:.4f}) {c.text[:240].strip()}...")
        console.print()

if __name__ == "__main__":
    main()

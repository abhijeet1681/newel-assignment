from __future__ import annotations

import argparse
from pathlib import Path
from src.rag.index import build_index

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True, help="Path to Swiggy Annual Report PDF")
    p.add_argument("--persist", default="chroma_db", help="Directory to persist Chroma DB")
    p.add_argument("--recreate", action="store_true", help="Recreate index from scratch")
    args = p.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    n = build_index(str(pdf_path), persist_dir=args.persist, recreate=args.recreate)
    print(f"✅ Indexed {n} chunks into {args.persist}/")

if __name__ == "__main__":
    main()

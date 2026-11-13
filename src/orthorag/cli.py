"""Command-line entry point: `orthorag <command>` (or `python -m orthorag <command>`)."""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="orthorag", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download public 510(k) summaries (orthopedic plates & screws)")
    f.add_argument("--limit", type=int, default=150, help="number of recent summaries (stage 1)")
    f.add_argument("--since", default="2018-01-01")
    f.add_argument("--predicates", type=int, default=150, help="max predicate devices to add (stage 2)")

    sub.add_parser("parse", help="extract text, sections and cited predicates")

    i = sub.add_parser("index", help="build BM25 + ChromaDB indexes")
    i.add_argument("--models", nargs="*", default=None)

    e = sub.add_parser("evaluate", help="run the predicate-retrieval benchmark")
    e.add_argument("--max-queries", type=int, default=None)

    d = sub.add_parser("draft", help="draft 510(k) summary sections for a device description")
    d.add_argument("description", help="device description text, or @path/to/file.txt")

    args = p.parse_args(argv)

    if args.cmd == "fetch":
        from .fetch import fetch_predicates, fetch_seed
        from .parse import parse_all
        meta = fetch_seed(args.limit, args.since)
        print(f"seed: {len(meta)} documents")
        if args.predicates:
            parse_all()
            meta = fetch_predicates(args.predicates)
            print(f"with predicates: {len(meta)} documents")
    elif args.cmd == "parse":
        from .parse import parse_all
        print(f"parsed: {parse_all()} documents")
    elif args.cmd == "index":
        from .index import build_index
        print(f"indexed: {build_index(args.models)} chunks")
    elif args.cmd == "evaluate":
        from .evaluate import run_benchmark
        print(run_benchmark(args.max_queries).round(3).to_string())
    elif args.cmd == "draft":
        from .generate import draft
        text = args.description
        if text.startswith("@"):
            with open(text[1:]) as fh:
                text = fh.read()
        for s in draft(text):
            print(f"\n## {s.section}\n{s.text}\n")
            print(f"sources: {', '.join(h.k_number for h in s.sources)}")
            if s.invalid_citations:
                print(f"⚠ citations not in sources: {', '.join(s.invalid_citations)}")


if __name__ == "__main__":
    main()

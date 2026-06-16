from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.retriever import Chunk, HybridRetriever, save_chunks_jsonl


TEXT_FIELDS = [
    "overview",
    "symptoms",
    "diagnosis",
    "treatment",
    "prognosis",
    "etiology",
    "prevention",
    "epidemiology",
]

NOISE_PATTERNS = [
    r"热门标签[:：].*",
    r"Telephone[:：].*",
    r"Email[:：].*",
    r".*ICP备.*",
    r".*公网安备.*",
]


def _load_csv(csv_path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return pd.read_csv(
                csv_path,
                engine="python",
                on_bad_lines="skip",
                encoding=encoding,
            )
        except Exception:
            continue
    raise RuntimeError(f"Unable to parse CSV: {csv_path}")


def _clean_text(raw_text: str) -> str:
    text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text or text.lower() == "nan":
        return ""

    clean_lines: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in NOISE_PATTERNS):
            continue
        clean_lines.append(line)

    deduped_lines: list[str] = []
    seen: set[str] = set()
    for line in clean_lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_lines.append(line)

    return "\n".join(deduped_lines).strip()


def build_chunks(df: pd.DataFrame) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, row in df.iterrows():
        doc_id = str(row.get("id", i))
        title = str(row.get("name", "") or row.get("english_name", "")).strip()
        source_url = str(row.get("url", "")).strip()
        aliases = str(row.get("aliases", "")).strip()

        header = f"Disease: {title}\nAliases: {aliases}\n"
        for field in TEXT_FIELDS:
            text = _clean_text(str(row.get(field, "")))
            if not text:
                continue
            merged = f"{header}\nSection: {field}\n{text}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_{field}",
                    doc_id=doc_id,
                    title=title,
                    field=field,
                    text=merged[:2500],
                    source_url=source_url,
                )
            )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to source CSV")
    parser.add_argument("--out", type=str, default="data", help="Output data directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_csv(csv_path)
    chunks = build_chunks(df)
    retriever = HybridRetriever(chunks)

    save_chunks_jsonl(chunks, out_dir / "chunks.jsonl")
    retriever.save(out_dir / "retriever.pkl")
    print(f"Loaded rows: {len(df)}")
    print(f"Built chunks: {len(chunks)}")
    print(f"Saved: {out_dir / 'chunks.jsonl'} and {out_dir / 'retriever.pkl'}")


if __name__ == "__main__":
    main()

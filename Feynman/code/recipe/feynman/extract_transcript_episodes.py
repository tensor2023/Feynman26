"""Extract complete, provenance-rich episodes from cleaned interview transcripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


YEARS = {
    "fun_to_imagine_complete_1983.txt": 1983,
    "take_the_world_1973.txt": 1973,
    "why_questions_transcript.txt": 1983,
    "blank_on_blank_1966.txt": 1966,
}


def clean(text: str) -> str:
    text = re.sub(r"^\[\d\d:\d\d:\d\d(?:\.\d+)?\]\s*", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunks(text: str, low: int = 130, high: int = 285) -> list[str]:
    parts = [clean(p) for p in re.split(r"\n\s*\n|(?=(?:Interviewer|Feynman):)", text) if clean(p)]
    result: list[str] = []
    current: list[str] = []
    words = 0
    for part in parts:
        count = len(part.split())
        if current and words + count > high:
            if words >= low:
                result.append(" ".join(current))
            current, words = [], 0
        current.append(part)
        words += count
        if words >= low:
            result.append(" ".join(current))
            current, words = [], 0
    if current and words >= low:
        result.append(" ".join(current))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows: list[dict] = []
    for path in sorted(args.input_dir.glob("*.txt")):
        if path.name not in YEARS:
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        body = re.sub(r"\A.*?(?=\n\s*\n)", "", raw, count=1, flags=re.S)
        for index, episode in enumerate(chunks(body), 1):
            # Remove obvious web chrome and community comments.
            if any(x in episode[:180] for x in ("LESSWRONG LW", "Supported By", "Thank you for the transcript")):
                continue
            digest = hashlib.sha256(episode.encode("utf-8")).hexdigest()
            source_id = path.stem.upper().replace("-", "_")
            rows.append({
                "episode_id": f"TXT-{source_id}-E{index:04d}",
                "source_id": f"TXT-{source_id}",
                "title": path.stem.replace("_", " ").title(),
                "source_year": YEARS[path.name],
                "page": index,
                "text": episode,
                "word_count": len(episode.split()),
                "quality_score": 1.0,
                "source_sha256": digest,
                "local_path": str(path),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} transcript episodes to {args.output}")


if __name__ == "__main__":
    main()

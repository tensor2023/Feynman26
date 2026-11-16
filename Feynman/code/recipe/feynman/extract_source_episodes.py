"""Extract candidate reasoning episodes from locally collected primary sources.

The extractor is deterministic and intentionally conservative: it uses only PDFs
with embedded text, keeps page provenance, repairs line-wrap hyphenation, and
groups complete sentences into moderate-size source episodes.  It does not call
an LLM; generation and verification happen in later, separately cached stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass(frozen=True)
class Source:
    source_id: str
    title: str
    year: int
    relative_path: str


SOURCES = (
    Source("FA1956L", "Letters", 1956, "feynman_archive/talks/1956_letters.pdf"),
    Source("FA1957H", "An Historic Moment in Physics", 1957, "feynman_archive/talks/1957_an_historic_moment_in_physics.pdf"),
    Source("FA1965M", "New Textbooks for the New Mathematics", 1965, "feynman_archive/talks/1965_new_textbooks_for_new_mathematics.pdf"),
    Source("FA1987W", "Mr. Feynman Goes to Washington", 1987, "feynman_archive/talks/1987_mr_feynman_goes_to_washington.pdf"),
    Source("FA1988S", "It's as Simple as One, Two, Three", 1988, "feynman_archive/talks/1988_its_as_simple_as_one_two_three.pdf"),
    Source("F020", "Simulating Physics with Computers", 1982, "feynman/F020_simulating_physics_with_computers.pdf"),
    Source("F023", "Personal Observations on Reliability of Shuttle", 1986, "feynman/F023_challenger_appendix_f.html"),
    Source("FASI", "Feynman Lectures on the Strong Interactions", 1987, "feynman_archive/strong_interactions/feynman_lectures_on_strong_interactions.pdf"),
)

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
NOISE = re.compile(r"^(?:\d+|references?|figure\s+\d+|table\s+\d+)$", re.I)
SIGNAL_WORDS = re.compile(
    r"\b(?:because|therefore|experiment|measure|means|suppose|example|problem|"
    r"question|result|physics|mathemat|probab|computer|particle|energy|theory|"
    r"evidence|calculate|understand|reason|model|equation|test|observe)\w*\b",
    re.I,
)


def clean_page(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line or NOISE.fullmatch(line):
            continue
        if re.fullmatch(r"[\d\s.,+\-=/()A-Za-z]{1,18}", line) and not re.search(r"[.!?]", line):
            continue
        lines.append(line)
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentence_chunks(text: str, minimum: int = 110, maximum: int = 260) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_END.split(text) if len(s.split()) >= 4]
    chunks: list[str] = []
    current: list[str] = []
    words = 0
    for sentence in sentences:
        count = len(sentence.split())
        if current and words + count > maximum:
            if words >= minimum:
                chunks.append(" ".join(current))
            current, words = [], 0
        current.append(sentence)
        words += count
        if words >= minimum and sentence.endswith((".", "?", "!")):
            chunks.append(" ".join(current))
            current, words = [], 0
    if current and words >= minimum:
        chunks.append(" ".join(current))
    return chunks


def quality_score(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    signals = len(SIGNAL_WORDS.findall(text))
    sentences = max(1, len(SENTENCE_END.split(text)))
    first_person = len(re.findall(r"\b(?:I|we|my|our)\b", text))
    equation_noise = len(re.findall(r"[=<>]|\b(?:Fig\.|Eq\.|Table)\b", text))
    return signals / len(words) + min(first_person, 5) * 0.002 + min(sentences, 10) * 0.001 - equation_noise * 0.002


def extract(data_files: Path, destination: Path, limit: int) -> int:
    rows = []
    for source in SOURCES:
        path = data_files / source.relative_path
        if path.suffix.lower() == ".html":
            soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
            raw_text = " ".join(soup.stripped_strings)
            sections = []
            for part in re.split(r"\[\s*F(\d+)\s*\]", raw_text):
                if part.isdigit():
                    sections.append((int(part), ""))
                elif sections:
                    number, previous = sections[-1]
                    sections[-1] = (number, previous + " " + part)
            pages = sections or [(1, raw_text)]
        else:
            reader = PdfReader(str(path))
            pages = [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]
        for page_number, raw_page_text in pages:
            page_text = clean_page(raw_page_text)
            for chunk_number, text in enumerate(sentence_chunks(page_text), 1):
                if "Richard P. Feynman" not in text and len(SIGNAL_WORDS.findall(text)) < 2:
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                rows.append(
                    {
                        "episode_id": f"{source.source_id}-P{page_number:03d}-E{chunk_number:02d}",
                        "source_id": source.source_id,
                        "title": source.title,
                        "source_year": source.year,
                        "page": page_number,
                        "text": text,
                        "word_count": len(text.split()),
                        "quality_score": round(quality_score(text), 6),
                        "source_sha256": digest,
                        "local_path": str(path.resolve()),
                    }
                )
    rows.sort(key=lambda row: (-row["quality_score"], row["source_id"], row["page"], row["episode_id"]))
    # Limit domination by the long strong-interactions lecture while retaining diversity.
    selected, counts = [], {}
    per_source_cap = max(20, limit // 2)
    for row in rows:
        count = counts.get(row["source_id"], 0)
        if count >= per_source_cap:
            continue
        selected.append(row)
        counts[row["source_id"]] = count + 1
        if len(selected) >= limit:
            break
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_files", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--limit", type=int, default=140)
    args = parser.parse_args()
    count = extract(args.data_files, args.destination, args.limit)
    print(f"Wrote {count} source episodes to {args.destination}")


if __name__ == "__main__":
    main()

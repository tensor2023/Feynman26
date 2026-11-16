"""Run cached Codex CLI prompts concurrently without using an API."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("schema", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="gpt-5.6-sol")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def run(prompt_path: Path) -> tuple[str, int]:
        output = args.output_dir / f"{prompt_path.stem}.json"
        if output.exists() and output.stat().st_size:
            return prompt_path.name, 0
        command = [
            shutil.which("codex.cmd") or shutil.which("codex") or "codex", "exec", "--ephemeral", "--skip-git-repo-check",
            "--sandbox", "read-only", "--model", args.model,
            "-c", 'model_reasoning_effort="low"',
            "--output-schema", str(args.schema), "-o", str(output), "-",
        ]
        completed = subprocess.run(
            command, input=prompt_path.read_text(encoding="utf-8"),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace",
        )
        (log_dir / f"{prompt_path.stem}.log").write_text(completed.stdout, encoding="utf-8")
        return prompt_path.name, completed.returncode

    prompts = sorted(args.prompt_dir.glob("generation_*.txt"))
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, prompt) for prompt in prompts]
        for done, future in enumerate(as_completed(futures), 1):
            name, code = future.result()
            failures += code != 0
            print(f"[{done}/{len(futures)}] {name}: {'ok' if code == 0 else f'failed ({code})'}", flush=True)
    if failures:
        raise SystemExit(f"{failures} batch(es) failed")


if __name__ == "__main__":
    main()

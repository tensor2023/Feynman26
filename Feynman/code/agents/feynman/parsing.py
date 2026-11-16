"""Pure parsing helpers shared by rollout code and dependency-light tests."""

from __future__ import annotations

import json
import re


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def parse_action(text: str | None) -> tuple[list[str], str]:
    clean = strip_think(text or "")
    try:
        obj = json.loads(clean)
        ids = obj.get("evidence_ids", [])
        response = obj.get("response", "")
        if isinstance(ids, list) and isinstance(response, str):
            return [str(item) for item in ids], response.strip()
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return [], clean

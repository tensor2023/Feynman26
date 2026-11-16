#!/usr/bin/env python3
"""Make one minimal inference request and fail closed if the judge cannot bill."""

import asyncio
import os
import sys

from openai import APIStatusError
from agents.utils import _get_openai_client, get_judge_model


async def probe() -> None:
    client = _get_openai_client()
    try:
        await client.responses.create(
            model=get_judge_model(os.environ["FEYNMAN_JUDGE_MODEL"]),
            input="Reply OK.",
            max_output_tokens=16,
        )
    finally:
        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(probe())
    except APIStatusError as exc:
        print(f"Judge billing probe failed: HTTP {exc.status_code}.", file=sys.stderr)
        raise SystemExit(1) from None
    print("Judge billing probe succeeded.")

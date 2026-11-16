#!/usr/bin/env python3
"""Export a PEFT LoRA adapter from a verl FSDP actor checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    actor_dir = checkpoint / "actor" if (checkpoint / "actor").is_dir() else checkpoint
    output = (args.output or actor_dir / "lora_adapter").resolve()
    state_path = actor_dir / "model_world_size_1_rank_0.pt"
    config_path = actor_dir / "lora_adapter" / "adapter_config.json"

    state = torch.load(state_path, map_location="cpu", weights_only=False, mmap=True)
    adapter = {
        key.replace(".default.weight", ".weight"): value.detach().contiguous().cpu()
        for key, value in state.items()
        if ".lora_A.default.weight" in key or ".lora_B.default.weight" in key
    }
    if not adapter:
        raise RuntimeError(f"No LoRA tensors found in {state_path}")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    # Parse before writing so a truncated/malformed config cannot produce a
    # seemingly valid but unusable adapter directory.
    with config_path.open(encoding="utf-8") as handle:
        json.load(handle)

    output.mkdir(parents=True, exist_ok=True)
    save_file(adapter, output / "adapter_model.safetensors")
    if config_path.resolve() != (output / "adapter_config.json").resolve():
        (output / "adapter_config.json").write_bytes(config_path.read_bytes())
    size = (output / "adapter_model.safetensors").stat().st_size
    print(f"Exported {len(adapter)} LoRA tensors ({size:,} bytes) to {output}")


if __name__ == "__main__":
    main()

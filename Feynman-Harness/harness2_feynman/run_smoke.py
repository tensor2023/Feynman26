from pathlib import Path

from harness2.core.proposer import Proposer
from harness2.core.schedule import run_task

from feynman_adapter import DeepSeekProposer, FeynmanAdapter, ROOT


def main() -> None:
    adapter = FeynmanAdapter()
    task_id = adapter.tasks("optimization")[0]
    proposer = DeepSeekProposer(adapter.spec.channels["proposer"], audit_dir=ROOT / "runs" / "proposer-audit")
    result = run_task(
        task_id=task_id,
        k=1,
        adapter=adapter,
        prop=proposer,
        selector_prop=proposer,
        runs_root=adapter.spec.runs_root,
        modes=("seq",),
        seed=42,
        tag="smoke2",
    )
    print(result.to_dict())


if __name__ == "__main__":
    main()

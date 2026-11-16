# Feynman26

Feynman26 is a source-grounded reinforcement-learning task for training and
evaluating documented scientific reasoning and explanatory behavior. It is not
a claim that a model is Richard Feynman, and it does not reward quotation
recall or superficial imitation.

The implementation adds:

- structured, historically bounded scenario and evidence schemas;
- independent multidimensional judging;
- correctness- and grounding-gated rewards;
- deterministic source-copy detection;
- verbal-feedback and refined FoldGRPO rollouts;
- dataset preparation, example scenarios, and core tests.

The Feynman26 implementation lives in `agents/feynman/` and
`recipe/feynman/`. See `recipe/feynman/README.md` for usage.

## Upstream foundation and attribution

Feynman26 is implemented on top of the Apache-2.0-licensed
[OdysSim](https://github.com/sunnweiwei/OdysSim) training framework by the
OdysSim authors. It is a derived project, not the upstream OdysSim repository.
The original upstream clone is kept separately at `../../ref/OdysSim`.

The upstream README is preserved as `UPSTREAM_ODYSSIM_README.md`, and the
upstream Apache-2.0 license remains in `LICENSE`.

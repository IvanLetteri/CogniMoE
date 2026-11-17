# Skill Profile: PROSE

- **Last Updated:** 2026-11-17 11:37
- **Target Hardware:** RTX 2070 Super (CUDA0 - 4800 MiB Cache)
- **Theoretical VRAM Hit Rate (Top-66/layer):** 78.70%
- **Telemetry File:** `profile_prose.json`

## Top-20 Global Hot Experts
`[107, 196, 35, 28, 147, 102, 43, 42, 47, 14, 229, 3, 110, 20, 106, 44, 155, 13, 233, 64]`

## Routing Notes
Pre-warming VRAM with `profile_prose.json` eliminates cold-start PCIe transfers for prose workloads.

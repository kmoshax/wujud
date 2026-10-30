# Code conventions

wujud is open source and is read by students who are learning what it does. Clarity beats cleverness.

## Layout
| Path | What lives there |
|---|---|
| `crates/proto` | Wire format between ESP32 boards and the host. `no_std`, shared with `firmware/`. |
| `crates/sim` | The 3-D physics simulator (rooms, halls, bodies, walls, ESP32 impairments) and dataset generation. |
| `crates/model` | Features from CSI, gradient-boosted trees (training + inference), the occupancy latch, room/hall/beacon engines. Pure logic, no I/O besides reading files. |
| `crates/app` | The `wujud` binary: CLI, live dashboard server, serial ports, laptop beacon capture, replay, train, bench. |
| `firmware/` | ESP32 firmware in Rust (separate Cargo project, targets the board). |
| `research/` | The original Python research code. It is the reference: Rust ports must reproduce its numbers. |

## Rules
1. **The code explains itself; comments are rare.** Names of types, functions and variables carry the meaning (`fresnel_radius`, `breathing_band_power`, not `f`, `bp`). Split long functions into well-named small ones instead of commenting sections. No comments that restate what the code does, no banner comments, no per-line narration.
   A comment is allowed only for a *why* the code cannot show: a physical reason, a measured number (with its source file), a deliberate deviation from the Python reference, or a non-obvious constraint. One short line, not a paragraph.
2. The physics explanations, module maps and the mapping to the Python reference (`research/...::function`) belong in the crate `README.md` / `docs/`, not in the code. A one-line `//!` at the top of a module is fine.
3. **Golden tests**: a Python script `research/tools/export_golden_<crate>.py` writes reference inputs/outputs to `crates/<crate>/tests/golden/` (small `.npy`/`.json`, keep the folder under ~5 MB); `crates/<crate>/tests/` compares against them with explicit tolerances. Run the Python with `research/.venv/bin/python` from inside `research/`.
4. Small, named functions over long ones; no macros beyond derives; no `unsafe`.
5. Physics in `f64`. Bulk sample storage may be `f32`/`i8`.
6. Randomness only through a seeded `rand_chacha::ChaCha8Rng` passed in explicitly, so every run is reproducible.
7. Dependencies only via `[workspace.dependencies]` in the root `Cargo.toml` (add new ones there).
8. `cargo fmt`, `cargo clippy --all-targets -- -D warnings` and `cargo test` pass for your crate before you are done.
9. Libraries return `Result<_, thiserror-error>`; only `crates/app` uses `anyhow`.
10. Measured claims in comments must match a file in `research/results/` or a test; never write a number you did not measure.

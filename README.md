# Ember Bullet V2 Trainer

Training and export tool for the **Ember NNUE v2** neural evaluation function based on the [bullet](https://github.com/jw1912/bullet) library. Trains the network and exports the weights into a binary container in Ember V2 format, which is read by the Ember engine.

- **Package:** `ember-bullet-v2-trainer`
- **Version:** 0.1.0
- **Edition:** 2024
- **License:** MIT
- **bullet repository:** https://github.com/jw1912/bullet

## What this program does

1. **Trains** the NNUE v2 architecture on chess data in `sfbinpack` (binpack) format — requires a GPU backend for bullet (CUDA/ROCm).
2. **Exports** weights from a checkpoint, or a network with random initialization (via `--init-seed`), into an Ember V2 format container:
   - container version `0x6A448AFA`,
   - architecture hash `0x0256ACDF`,
   - 8 stacks, Full Threats 60,720 inputs + HalfKAv2_hm 22,528 inputs,
   - weights quantized to `i8` / `i16` / `i32` depending on the layer.
3. **Verifies** the written container by reading it back and checking versions, hashes, sizes, and weight ranges.

## Quick start

```bash
# Build in release mode
cargo build --release

# CLI help
cargo run --release -- --help
```

The binary will be available as `target/release/ember-bullet-v2-trainer` (or via `cargo run --release`).

## CLI

The program accepts one of two subcommands:

```
ember-bullet-v2-trainer <command>
```

### `train` — training

Trains the network on binpack data. Requires building bullet with a CUDA/ROCm backend; in the current build without a GPU backend the command errors out and suggests using `export` instead.

```bash
cargo run --release -- train <PATH_TO_DATA> \
  --batch-size 4096 \
  --lr 1e-3 \
  --superbatches 10 \
  --out-dir checkpoints \
  --load-checkpoint checkpoints/last.pt \
  --lambda 1.0 \
  --smoke
```

| Argument | Default | Description |
|---|---|---|
| `<DATA>` | — | Path to the binpack training data file |
| `--batch-size` | 4096 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--superbatches` | 10 | Number of superbatches |
| `--out-dir` | `checkpoints` | Directory for checkpoints |
| `--load-checkpoint` | — | Path to checkpoint to resume training |
| `--lambda` | 1.0 | Lambda coefficient (mixing of evaluation and result) |
| `--smoke` | — | Quick check mode (smoke test) |

### `export` — Ember V2 container export

Exports weights from a checkpoint (or creates a network with random weights via `--init-seed`) into a binary Ember V2 format container.

```bash
# From a checkpoint
cargo run --release -- export \
  --checkpoint checkpoints/final.pt \
  --out ember_nnue_v2.bin \
  --description "Ember NNUE v2 trained with bullet"

# Random initialization (without checkpoint)
cargo run --release -- export \
  --init-seed 12345 \
  --out ember_nnue_v2_init.bin \
  --description "untrained init net"
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | — | Path to checkpoint with weights (required if no `--init-seed`) |
| `--out` | — | Path to output container (required) |
| `--description` | `"Ember NNUE v2 trained with bullet"` | Description written into the container |
| `--init-seed` | — | Seed for random weight initialization (required if no `--checkpoint`) |

Example successful output:

```
Exported 1234567 bytes to "ember_nnue_v2.bin" (arch 0x256acdf, stacks 8, description: "Ember NNUE v2 trained with bullet")
```

## Model

- `L1 = 1024`, `L2 = 32`, `L3 = 1`, `NUM_BUCKETS = 8`
- `FT_OUTPUTS = 1032` (1024 main + 8 bucket outputs)
- `THREAT_DIMS = 60 720`, `HALKKA_DIMS = 22 528`, `COMBINED_INPUTS = 83 248`
- Two full-threat inputs (`ft_stm`, `ft_ntm`), buffers `buckets`, `skip_idx_30`, `skip_idx_31`, dense inputs `us`, `score`, `outcome`, `lambda`
- Output is converted to centipawns (`cp = output * 600`), sigmoids `qf` and `pf`, loss function `|pt - qf|^2.5`

## Weight quantization

| Tensor | Type | Scale |
|---|---|---|
| `ft_w`, `ft_b` | i8 / i16 | 255 / 256 |
| `l1/w`, `l1/b` | i8 / i32 | 128 / 16384 |
| `l2/w`, `l2/b` | i8 / i32 | 64 / 8192 |
| `out/w`, `out/b` | i8 / i32 | 128 / 16384 |

## Ember V2 container format

The container is written sequentially:

1. `version` (u32 LE) = `0x6A448AFA`
2. `arch_hash` (u32 LE) = `0x0256ACDF`
3. `description` — length (u32 LE) + UTF-8 bytes
4. `ft_header_hash` (u32 LE) = `0x6165DDC9`
5. Block `ft_bias` — LEB128 compression (`COMPRESSED_LEB128`), scale 256
6. Block `threat_weights` — `i8`, scale 256, size `60 720 × 1 024`
7. Block `threat_psqt` — LEB128, scale 9600, size `60 720 × 8`
8. Block `psq_weights` — LEB128, scale 256, size `22 528 × 1 024`
9. Block `psqt` — LEB128, scale 9600, size `22 528 × 8`
10. 8 stacks, each starting with `stack_hash` = `0x63337116`, one stack size 35,208 bytes:
    - `fc0_bias` (32 × i32, scale 16384)
    - `fc0_weights` (32 × 1024 × i8, scale 128)
    - `fc1_bias` (32 × i32, scale 8192)
    - `fc1_weights` (32 × 64 × i8, scale 64)
    - `fc2_bias` (i32, scale 16384)
    - `fc2_weights` (128 × i8, scale 128)

After writing, the program reads the container back (`parse_container`) and verifies all hashes, versions, and sizes.

## Testing

```bash
cargo test
```

The test suite includes:

- Known LEB128 vectors (`write_sleb128` / `decode_signed_leb`)
- Round-trip container write and read (`container_self_test`)
- Model construction without training (`read_model_weights_compiles`)
- Bit-exact match of full-threat and HalfKAv2 indices.

## Dependencies

- `bullet_lib`, `bullet-compiler`, `bullet-trainer` — from the `jw1912/bullet` repository (fixed rev)
- `bulletformat` 1.8.0
- `sfbinpack` 0.6.4 — reading training binpack data
- `anyhow` 1 — error handling
- `clap` 4 (with `derive`) — CLI

## License

Distributed under the MIT license.
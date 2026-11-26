# Benchmark

GPU: NVIDIA GeForce RTX 5090  |  torch 2.15.0+cu132  |  CUDA 13.2

## Training step

| grid | network | batch | precision | s/step | peak GB | h for 10k / 40k / 100k steps |
|---|---|---|---|---|---|---|
| coarse | small | 8 | fp32 | 0.0331 | 0.47 | 0.09 / 0.37 / 0.92 |
| coarse | small | 16 | fp32 | 0.0558 | 0.87 | 0.16 / 0.62 / 1.55 |
| coarse | small | 32 | fp32 | 0.0939 | 1.66 | 0.26 / 1.04 / 2.61 |
| coarse | small | 8 | tf32 | 0.0285 | 0.46 | 0.08 / 0.32 / 0.79 |
| coarse | small | 16 | tf32 | 0.0296 | 0.87 | 0.08 / 0.33 / 0.82 |
| coarse | small | 32 | tf32 | 0.0584 | 1.67 | 0.16 / 0.65 / 1.62 |
| coarse | small | 8 | bf16 | 0.0294 | 0.42 | 0.08 / 0.33 / 0.82 |
| coarse | small | 16 | bf16 | 0.0416 | 0.77 | 0.12 / 0.46 / 1.16 |
| coarse | small | 32 | bf16 | 0.0771 | 1.45 | 0.21 / 0.86 / 2.14 |
| coarse | medium | 8 | fp32 | 0.0606 | 0.81 | 0.17 / 0.67 / 1.68 |
| coarse | medium | 16 | fp32 | 0.088 | 1.47 | 0.24 / 0.98 / 2.44 |
| coarse | medium | 32 | fp32 | 0.1329 | 2.81 | 0.37 / 1.48 / 3.69 |
| coarse | medium | 8 | tf32 | 0.0331 | 0.8 | 0.09 / 0.37 / 0.92 |
| coarse | medium | 16 | tf32 | 0.0558 | 1.47 | 0.16 / 0.62 / 1.55 |
| coarse | medium | 32 | tf32 | 0.0949 | 2.81 | 0.26 / 1.05 / 2.64 |
| coarse | medium | 8 | bf16 | 0.0303 | 0.72 | 0.08 / 0.34 / 0.84 |
| coarse | medium | 16 | bf16 | 0.0313 | 1.28 | 0.09 / 0.35 / 0.87 |
| coarse | medium | 32 | bf16 | 0.0698 | 2.41 | 0.19 / 0.78 / 1.94 |
| coarse | large | 8 | fp32 | 0.1094 | 1.61 | 0.3 / 1.22 / 3.04 |
| coarse | large | 16 | fp32 | 0.1869 | 2.83 | 0.52 / 2.08 / 5.19 |
| coarse | large | 32 | fp32 | 0.328 | 5.23 | 0.91 / 3.64 / 9.11 |
| coarse | large | 8 | tf32 | 0.0626 | 1.61 | 0.17 / 0.7 / 1.74 |
| coarse | large | 16 | tf32 | 0.097 | 2.82 | 0.27 / 1.08 / 2.69 |
| coarse | large | 32 | tf32 | 0.1689 | 5.22 | 0.47 / 1.88 / 4.69 |
| coarse | large | 8 | bf16 | 0.0406 | 1.51 | 0.11 / 0.45 / 1.13 |
| coarse | large | 16 | bf16 | 0.064 | 2.6 | 0.18 / 0.71 / 1.78 |
| coarse | large | 32 | bf16 | 0.1075 | 4.77 | 0.3 / 1.19 / 2.99 |
| fine | small | 2 | fp32 | 0.0599 | 0.82 | 0.17 / 0.67 / 1.66 |
| fine | small | 4 | fp32 | 0.089 | 1.52 | 0.25 / 0.99 / 2.47 |
| fine | small | 8 | fp32 | 0.1475 | 2.98 | 0.41 / 1.64 / 4.1 |
| fine | small | 2 | tf32 | 0.0644 | 0.81 | 0.18 / 0.72 / 1.79 |
| fine | small | 4 | tf32 | 0.0401 | 1.52 | 0.11 / 0.45 / 1.11 |
| fine | small | 8 | tf32 | 0.0846 | 2.98 | 0.23 / 0.94 / 2.35 |
| fine | small | 2 | bf16 | 0.0351 | 0.72 | 0.1 / 0.39 / 0.98 |
| fine | small | 4 | bf16 | 0.0591 | 1.32 | 0.16 / 0.66 / 1.64 |
| fine | small | 8 | bf16 | 0.0962 | 2.58 | 0.27 / 1.07 / 2.67 |
| fine | medium | 2 | fp32 | 0.0918 | 1.39 | 0.26 / 1.02 / 2.55 |
| fine | medium | 4 | fp32 | 0.1353 | 2.58 | 0.38 / 1.5 / 3.76 |
| fine | medium | 8 | fp32 | 0.2541 | 5.01 | 0.71 / 2.82 / 7.06 |
| fine | medium | 2 | tf32 | 0.0556 | 1.39 | 0.15 / 0.62 / 1.54 |
| fine | medium | 4 | tf32 | 0.0858 | 2.57 | 0.24 / 0.95 / 2.38 |
| fine | medium | 8 | tf32 | 0.1377 | 5.02 | 0.38 / 1.53 / 3.83 |
| fine | medium | 2 | bf16 | 0.0305 | 1.21 | 0.08 / 0.34 / 0.85 |
| fine | medium | 4 | bf16 | 0.0574 | 2.22 | 0.16 / 0.64 / 1.59 |
| fine | medium | 8 | bf16 | 0.0904 | 4.27 | 0.25 / 1.0 / 2.51 |
| fine | large | 2 | fp32 | 0.1844 | 2.64 | 0.51 / 2.05 / 5.12 |
| fine | large | 4 | fp32 | 0.3192 | 4.81 | 0.89 / 3.55 / 8.87 |
| fine | large | 8 | fp32 | 0.5825 | 9.22 | 1.62 / 6.47 / 16.18 |
| fine | large | 2 | tf32 | 0.0926 | 2.64 | 0.26 / 1.03 / 2.57 |
| fine | large | 4 | tf32 | 0.1555 | 4.81 | 0.43 / 1.73 / 4.32 |
| fine | large | 8 | tf32 | 0.2916 | 9.34 | 0.81 / 3.24 / 8.1 |
| fine | large | 2 | bf16 | 0.0626 | 2.44 | 0.17 / 0.7 / 1.74 |
| fine | large | 4 | bf16 | 0.0961 | 4.41 | 0.27 / 1.07 / 2.67 |
| fine | large | 8 | bf16 | 0.1673 | 8.4 | 0.46 / 1.86 / 4.65 |

## One solver step (batch 1, no gradients)

| grid | network | precision | s/step | steps/s |
|---|---|---|---|---|
| coarse | small | fp32 | 0.0019 | 525.9 |
| coarse | small | tf32 | 0.00086 | 1167.5 |
| coarse | small | bf16 | 0.00116 | 860.5 |
| coarse | medium | fp32 | 0.00388 | 257.7 |
| coarse | medium | tf32 | 0.00115 | 870.7 |
| coarse | medium | bf16 | 0.0012 | 832.7 |
| coarse | large | fp32 | 0.01354 | 73.9 |
| coarse | large | tf32 | 0.00241 | 414.2 |
| coarse | large | bf16 | 0.00134 | 747.7 |
| fine | small | fp32 | 0.0039 | 256.3 |
| fine | small | tf32 | 0.00346 | 288.8 |
| fine | small | bf16 | 0.00135 | 741.5 |
| fine | medium | fp32 | 0.01017 | 98.3 |
| fine | medium | tf32 | 0.00348 | 286.9 |
| fine | medium | bf16 | 0.00261 | 383.4 |
| fine | large | fp32 | 0.0325 | 30.8 |
| fine | large | tf32 | 0.01237 | 80.8 |
| fine | large | bf16 | 0.00628 | 159.4 |

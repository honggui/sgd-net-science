# 8. From Computational Graph to Chip

For this simplified configuration, the four layers of BF16 z states at N=128 require 4×128×128×2=128 KiB; the real trajectory plus two candidates require 384 KiB. h, vectors, edges, coordinates, weights, optimizers, and all rollout intermediates are counted separately. Standard Mamba's additional state dimensions would change this result.

Matrix units handle projections and MLPs; Vector/Scan handles temporal operations and reductions; Sparse/Vector handles graph indexing; DMA handles bounded state transfer. Total cost includes graph construction, automatic differentiation, periodic neighborhoods, and tool waiting. Updating computational coordinates is not committing a device action; hardware runtimes do not interpret biological mechanisms.

By default, validate on general-purpose CPUs/GPUs before testing Aegis-X. Structural heads may need FP32 automatic differentiation and geometric accumulation; a BF16 backbone does not imply every operator can run at low precision. New hardware units and fixed memory figures require support from actual operator traces.


---

[← Previous](section-08.md) · [Contents](../../../SUMMARY.md) · [Next →](section-10.md)

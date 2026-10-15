# Settings shared by every benchmark in the paper (also the defaults of run.py).
# VGGT-Bridge = SwiftVGGT + coarse-stride skip edges (strides 3, 5)
#             + loop-aware reverse anchoring of the stride-3 chunks.
BRIDGE_ARGS="--coarse_strides 3,5 \
  --skip_edge_weight 0.3 \
  --adaptive_strides 3 \
  --loop_density_threshold 0.7 \
  --loop_density_window 60 \
  --robust_kernel cauchy --robust_delta 2.0"
CKPT=${CKPT:-./ckpt/model_tracker_fixed_e20.pt}

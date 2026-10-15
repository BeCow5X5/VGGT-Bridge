#!/usr/bin/env bash
# Waymo Open, front camera. Shorter segments (~199 frames) use smaller chunks.
# Usage: bash scripts/run_waymo.sh <waymo_root> <output_root>
#   <waymo_root>/<segment>/image_0/*.png
#   <waymo_root>/<segment>/poses_camera.txt   (camera-to-world, KITTI format)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
ROOT=$1; OUT=$2
for seg in $(ls $ROOT | grep '^segment-'); do
  python run.py --image_dir $ROOT/$seg/image_0 --gt_pose_path $ROOT/$seg/poses_camera.txt \
    --output_path $OUT/$seg --ckpt $CKPT \
    --chunk_size 60 --overlap_size 24 \
    --coarse_chunk_size_s3 35 --coarse_overlap_size_s3 14 --coarse_chunk_size_s5 25 --coarse_overlap_size_s5 10 \
    $BRIDGE_ARGS
done

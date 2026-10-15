#!/usr/bin/env bash
# KITTI Odometry 00-10.
# Usage: bash scripts/run_kitti.sh <kitti_root> <output_root>
#   <kitti_root>/sequences/<seq>/image_2/*.png
#   <kitti_root>/poses/<seq>.txt
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
ROOT=$1; OUT=$2
for s in 00 01 02 03 04 05 06 07 08 09 10; do
  python run.py --image_dir $ROOT/sequences/$s/image_2 --gt_pose_path $ROOT/poses/$s.txt \
    --output_path $OUT/$s --ckpt $CKPT \
    --chunk_size 75 --overlap_size 30 $BRIDGE_ARGS
done

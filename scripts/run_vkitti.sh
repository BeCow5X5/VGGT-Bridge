#!/usr/bin/env bash
# Virtual KITTI 1.3.1, 5 scenes x 6 conditions. The paper reports the mean over conditions per scene.
# Usage: bash scripts/run_vkitti.sh <vkitti_root> <output_root>
#   <vkitti_root>/<Scene>/<cond>/frames/rgb/Camera_0/*.jpg
#   <vkitti_root>/<Scene>/<cond>/poses_cam0_kitti_inv.txt   (camera-to-world, KITTI format)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
ROOT=$1; OUT=$2
for sc in Scene01 Scene02 Scene06 Scene18 Scene20; do
  for c in clone fog morning overcast rain sunset; do
    python run.py --image_dir $ROOT/$sc/$c/frames/rgb/Camera_0 \
      --gt_pose_path $ROOT/$sc/$c/poses_cam0_kitti_inv.txt \
      --output_path $OUT/${sc}_${c} --ckpt $CKPT \
      --chunk_size 75 --overlap_size 30 $BRIDGE_ARGS
  done
done

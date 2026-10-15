"""Translation ATE (RMSE, m) after Sim(3) alignment, as reported in the paper.

Usage:
  python scripts/eval_ate.py --gt <gt_poses.txt> --est <output>/camera_poses.txt [--full]
Both files are in KITTI format (3x4 camera-to-world per line).
"""
import argparse
import numpy as np
import evo.main_ape as main_ape
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface


def ate(gt_path, est_path, pose_relation=PoseRelation.translation_part):
    gt = file_interface.read_kitti_poses_file(gt_path)
    est = file_interface.read_kitti_poses_file(est_path)
    traj_gt = PoseTrajectory3D(poses_se3=gt.poses_se3, timestamps=np.arange(gt.num_poses, dtype=float))
    traj_est = PoseTrajectory3D(poses_se3=est.poses_se3, timestamps=np.arange(est.num_poses, dtype=float))
    traj_gt, traj_est = sync.associate_trajectories(traj_gt, traj_est)
    result = main_ape.ape(traj_gt, traj_est, pose_relation=pose_relation,
                          align=True, correct_scale=True)
    return result.stats["rmse"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--est", required=True)
    parser.add_argument("--full", action="store_true", help="Also report the full-transformation ATE.")
    args = parser.parse_args()
    print(f"ATE (translation, Sim(3)-aligned) RMSE: {ate(args.gt, args.est):.3f} m")
    if args.full:
        print(f"ATE (full transformation, Sim(3)-aligned) RMSE: "
              f"{ate(args.gt, args.est, PoseRelation.full_transformation):.3f}")

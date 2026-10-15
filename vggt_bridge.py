# VGGT-Bridge: Beyond Sequential Pose Graphs via Coarse-Stride Skip Edges (ACCV 2026).
#
# This file is built on SwiftVGGT (https://github.com/Jho-Yonsei/SwiftVGGT, MIT License).
# The chunk-and-align pipeline, loop detection and loop-centric alignment come from SwiftVGGT.
# VGGT-Bridge adds the coarse-stride chunks, the skip edges, reverse anchoring with the
# loop-aware direction policy, and the weighted pose graph that optimizes all edges jointly.

import torch
import numpy as np
import trimesh
import scipy
import pypose as pp

import os
from glob import glob
from tqdm import tqdm
import time

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import load_images_rgb, get_vgg_input_imgs
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from utils.chunk_utils import *
from utils.loop_utils import *
from utils.loop_optimizer import *


class VGGTBridge:
    def __init__(self, args, device, dtype):
        self.args = args
        self.device = device
        self.dtype = dtype
        
        self.load_model(args.ckpt)
        self.load_images(args.image_dir)
        self.strides = self._parse_coarse_strides(args.coarse_strides)
        self.chunk_sets = self._build_chunk_sets(self.strides)
        self._set_fine_aliases()
        
        self.loop_optimizer = Sim3LoopOptimizer(args)
        self.loop_prediction_list = []
        self.loop_sim3_list = []
        self.skip_edges = []

    def _parse_coarse_strides(self, strides_arg):
        """Return the list of strides, with stride 1 (the fine chunks) always first."""
        if isinstance(strides_arg, (list, tuple)):
            parsed = [int(s) for s in strides_arg]
        else:
            parsed = [int(s.strip()) for s in str(strides_arg).split(',') if s.strip()]

        parsed = [s for s in parsed if s > 0]
        if not parsed:
            parsed = [1]

        if 1 in parsed:
            parsed = [1] + [s for s in parsed if s != 1]
        else:
            parsed = [1] + parsed

        unique = []
        for s in parsed:
            if s not in unique:
                unique.append(s)

        return unique


    def _get_chunk_params(self, stride):
        if stride == 1:
            return self.args.chunk_size, self.args.overlap_size
        # Explicit override: --coarse_chunk_size_s3 / --coarse_chunk_size_s5 take priority.
        cs_override = getattr(self.args, f"coarse_chunk_size_s{stride}", 0)
        os_override = getattr(self.args, f"coarse_overlap_size_s{stride}", 0)
        if cs_override and os_override:
            return int(cs_override), int(os_override)
        # Scale coarse chunk size proportionally to the fine chunk size
        # (stride 3: 35/75, stride 5: 25/75 of the fine size; overlap likewise).
        # For chunk_size=75 this gives (35, 14) and (25, 10).
        base_cs = self.args.chunk_size
        base_os = self.args.overlap_size
        if stride == 3:
            return max(10, round(base_cs * 35 / 75)), max(4, round(base_os * 14 / 30))
        if stride == 5:
            return max(10, round(base_cs * 25 / 75)), max(4, round(base_os * 10 / 30))

        base_chunk = max(10, int(base_cs / max(stride, 1)))
        base_overlap = max(4, int(base_os / max(stride, 1)))
        return base_chunk, min(base_overlap, max(base_chunk - 1, 1))


    def _build_chunk_sets(self, strides):
        # Coarse strides whose chunks are all reverse-anchored.
        reverse_strides_set = set()
        if self.args.reverse_strides:
            reverse_strides_set = {int(s) for s in self.args.reverse_strides.split(",") if s.strip()}

        # Coarse strides whose direction is decided per chunk by the loop-aware policy.
        adaptive_set = set()
        if self.args.adaptive_strides:
            adaptive_set = {int(s) for s in self.args.adaptive_strides.split(",") if s.strip()}

        chunk_sets = []
        for stride in strides:
            chunk_size, overlap_size = self._get_chunk_params(stride)
            sampled_indices = list(range(0, len(self.image_paths), stride))
            num_chunks, chunk_list = self.get_idx_list(
                chunk_size=chunk_size,
                overlap_size=overlap_size,
                total_len=len(sampled_indices),
            )

            chunk_sets.append({
                "stride": stride,
                "reversed": (stride > 1 and stride in reverse_strides_set),
                "adaptive": (stride > 1 and stride in adaptive_set),
                "sampled_indices": sampled_indices,
                "num_chunks": num_chunks,
                "chunk_list": chunk_list,
                "chunk_size": chunk_size,
                "overlap_size": overlap_size,
                "temporal_chunk_list": [],
                "sim3_list": [],
                "camera_extrinsics": [],
                "camera_intrinsics": [],
                "intrinsic_ref": None,
                "frame_to_chunk": {},
                "node_offset": 0,
            })

        offset = 0
        for chunk_set in chunk_sets:
            chunk_set["node_offset"] = offset
            offset += chunk_set["num_chunks"]

        self.total_nodes = offset

        return chunk_sets


    def _set_fine_aliases(self):
        base = self.chunk_sets[0]
        self.num_chunks = base["num_chunks"]
        self.chunk_list = base["chunk_list"]
        self.temporal_chunk_list = base["temporal_chunk_list"]
        self.sim3_list = base["sim3_list"]
        self.camera_extrinsics = base["camera_extrinsics"]
        self.camera_intrinsics = base["camera_intrinsics"]
        
        
    @torch.no_grad()
    def run_loop_detection(self, args):
        num_chunks_loop, chunk_list_loop = self.get_idx_list(chunk_size=64, overlap_size=0)
        patch_tokens_list = []
        print("Images to DINO Patch Tokens")
        for batch_idx, (start, end) in enumerate(chunk_list_loop):
            print(f"\tProcessing batch {batch_idx + 1:3d}/{num_chunks_loop}: images {start:4d} to {end - 1:4d}", end='\r', flush=True)
            
            patch_tokens = self.model.aggregator.forward_DINOv2(self.vgg_input[start: end].to(self.device, non_blocking=True))
            patch_tokens_list.append(patch_tokens)
            del patch_tokens
        print('\n')
        
        patch_tokens = torch.cat(patch_tokens_list, dim=0)
        del patch_tokens_list
        
        start = time.time()
        
        G = self.get_descriptors(patch_tokens, args)
        S_cos = cosine_sim_matrix(G)
        similarities, indices = S_cos.topk(k=args.topk)
        
        loop_closures = loop_detection(similarities, indices, 
                                       similarity_threshold=args.similarity_threshold,
                                       neighbor_threshold=args.neighbor_threshold)
        del similarities, indices, S_cos
        torch.cuda.empty_cache()
        
        loop_list = [(idx1, idx2) for idx1, idx2, _ in loop_closures]
        self.loop_frame_pairs = list(loop_list)
        half_window = getattr(self.args, "loop_half_window", 10)
        loop_results = process_loop_list(self.chunk_list, loop_list, half_window=half_window)
        loop_results = remove_duplicates(loop_results)
        
        end = time.time()
        
        return patch_tokens, loop_results
    
        
    def get_descriptors(self, patch_tokens, args):
        G0 = global_mean_pool(patch_tokens, chunk_size=args.pooling_batch_size)
        
        whitener = PCAWhitener(out_dim=args.pca_out_dim, remove_first_n=args.pca_remove_first_n)
        whitener.fit(G0)
        
        G = build_descriptors(patch_tokens, whitener=whitener, beta=args.signed_power_beta)
        
        return G
            
        
    def get_idx_list(self, chunk_size, overlap_size, total_len=None):
        step_size = chunk_size - overlap_size
        idx_list = []
        if total_len is None:
            total_len = len(self.image_paths)
        for start_idx in range(0, total_len, step_size):
            end_idx = start_idx + chunk_size
            if end_idx > total_len:
                end_idx = total_len
            idx_list.append([start_idx, end_idx])
            if end_idx == total_len:
                break
        return len(idx_list), idx_list
        
        
    def load_images(self, image_dir):
        self.image_paths = sorted([f for f in glob(os.path.join(image_dir, '*')) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        images = load_images_rgb(self.image_paths)
        images_array = np.stack(images)
        vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
        self.vgg_input = vgg_input  # CPU-resident; per-chunk slices are moved to the GPU at use
        self.model.update_patch_dimensions(patch_width, patch_height)
        
        print(f"Image patch dimensions: {patch_width}x{patch_height}", end='\n\n')
        
        
    def load_model(self, ckpt='./ckpt/model_tracker_fixed_e20.pt'):
        print(f"Loading model: '{ckpt}'")
        model = VGGT(merging=0, vis_attn_map=False)
        ckpt = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        model = model.cuda().eval()
        self.model = model.to(self.dtype)
        print("Model loaded")
        
    
    def run_single_temporal_chunk(self, set_idx, chunk_set, patch_tokens_chunk, images_chunk, chunk_idx, frame_ids, reversed_input=False):
        tokens_chunk, patch_start_idx = self.model.aggregator.forward_aggregator(patch_tokens_chunk, *images_chunk.shape[-2:])
        cam_tokens_chunk, patch_tokens_chunk = tokens_chunk[-1][:, :, 0:1, :], torch.stack(tokens_chunk)[:, :, :, 5:, :]
        prediction_chunk = self.model.run_heads(cam_tokens_chunk, patch_tokens_chunk, images_chunk, patch_start_idx)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(prediction_chunk['pose_enc'], images_chunk.shape[-2:])
        depth_map, depth_conf = prediction_chunk['depth'], prediction_chunk['depth_conf']
        del prediction_chunk

        extrinsic = extrinsic.squeeze(0).cpu().float().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().float().numpy()
        depth_map = depth_map.squeeze(0).cpu().float().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().float().numpy()

        if reversed_input:
            # VGGT was given frames in reverse temporal order, so its world frame is anchored at
            # the latest frame. Outputs are re-indexed to temporal order for the code downstream;
            # the poses stay in the reverse-anchored world frame, which is the point of reversal.
            extrinsic = extrinsic[::-1].copy()
            intrinsic = intrinsic[::-1].copy()
            depth_map = depth_map[::-1].copy()
            depth_conf = depth_conf[::-1].copy()
        
        if chunk_idx == 0:
            chunk_set["intrinsic_ref"] = intrinsic[0]
            if set_idx == 0:
                self.intrinsic_ref = chunk_set["intrinsic_ref"]
            
        intrinsic, depth_map = align_intrinsics_and_depth(intrinsic, chunk_set["intrinsic_ref"], depth_map[..., 0])
        points_3d = unproject_depth_map_to_point_map(depth_map[..., None], extrinsic, intrinsic)

        if chunk_idx != 0:
            chunk_list = chunk_set["chunk_list"]
            prev_start, prev_end = chunk_list[chunk_idx - 1]
            cur_start, cur_end = chunk_list[chunk_idx]
            prev_size = prev_end - prev_start
            actual_overlap = max(0, prev_end - cur_start)
            depth_map_prev = chunk_set["temporal_chunk_list"][-1]["prediction"]['depth_map']
            depth_conf_prev = chunk_set["temporal_chunk_list"][-1]["prediction"]['depth_conf']

            depth_map_prev_overlap = depth_map_prev[prev_size - actual_overlap:]
            depth_conf_prev_overlap = depth_conf_prev[prev_size - actual_overlap:]

            depth_map_cur_overlap = depth_map[:actual_overlap]
            depth_conf_cur_overlap = depth_conf[:actual_overlap]
            
            mask = (np.abs(depth_map_prev_overlap - depth_map_cur_overlap) < self.args.depth_diff_threshold) \
                * (depth_conf_prev_overlap > depth_conf_prev_overlap.mean(axis=(1, 2), keepdims=True) * self.args.conf_threshold) \
                * (depth_conf_cur_overlap > depth_conf_cur_overlap.mean(axis=(1, 2), keepdims=True) * self.args.conf_threshold)
                
            if mask.sum() < 20000:
                mask = (np.abs(depth_map_prev_overlap - depth_map_cur_overlap) < self.args.depth_diff_threshold + 0.05) \
                    * (depth_conf_prev_overlap > depth_conf_prev_overlap.mean(axis=(1, 2), keepdims=True) * self.args.conf_threshold) \
                    * (depth_conf_cur_overlap > depth_conf_cur_overlap.mean(axis=(1, 2), keepdims=True) * self.args.conf_threshold)
                    
            points_3d_prev = chunk_set["temporal_chunk_list"][-1]["prediction"]['points_3d']

            points_3d_prev_overlap = points_3d_prev[prev_size - actual_overlap:]
            points_3d_cur_overlap = points_3d[:actual_overlap]
            
            points_3d_prev_overlap = points_3d_prev_overlap[mask]
            points_3d_cur_overlap = points_3d_cur_overlap[mask]
            
            s, R, t = umeyama_sim3(points_3d_cur_overlap, points_3d_prev_overlap)
            chunk_set["sim3_list"].append((s, R, t))

        prediction_chunk = {'extrinsic': extrinsic, 'intrinsic': intrinsic, 'depth_map': depth_map, 
                            'depth_conf': depth_conf, 'points_3d': points_3d}
        
        chunk_range = (frame_ids[0], frame_ids[-1] + 1) if frame_ids else (0, 0)
        chunk_set["camera_extrinsics"].append((chunk_range, extrinsic))
        chunk_set["camera_intrinsics"].append((chunk_range, intrinsic))
        chunk_set["temporal_chunk_list"].append({
            "chunk_range": chunk_range,
            "frame_ids": frame_ids,
            "prediction": prediction_chunk,
        })
        
    
    def run_temporal_chunks(self, patch_tokens):
        print("Process Fine and Coarse Chunks")
        
        for set_idx, chunk_set in enumerate(self.chunk_sets):
            stride = chunk_set["stride"]
            num_chunks = chunk_set["num_chunks"]
            print(f"\tStride {stride}: {num_chunks} chunks")
            
            for chunk_idx in range(num_chunks):
                start_idx, end_idx = chunk_set["chunk_list"][chunk_idx]
                frame_ids = chunk_set["sampled_indices"][start_idx: end_idx]
                if len(frame_ids) == 0:
                    continue
                
                first_f = frame_ids[0]
                last_f = frame_ids[-1]
                print(
                    f"\t\tProcessing chunk {chunk_idx + 1:3d}/{num_chunks}: images {first_f:4d} to {last_f:4d}",
                    end='\r',
                    flush=True,
                )
                
                # The anchoring direction of a coarse chunk sets the sign of its scale bias.
                if chunk_set["adaptive"]:
                    # Loop-aware policy: chunks in drift regions (low loop density) are
                    # reverse-anchored, chunks near loop closures stay forward.
                    if not hasattr(self, "_loop_density_cache"):
                        self._loop_density_cache = self._compute_loop_density_array()
                    dens = self._loop_density_cache
                    chunk_density = float(np.mean(dens[np.asarray(frame_ids)]))
                    reverse_input = chunk_density < self.args.loop_density_threshold
                else:
                    reverse_input = chunk_set["reversed"]

                frame_ids_input = list(reversed(frame_ids)) if reverse_input else frame_ids

                frame_ids_tensor = torch.tensor(frame_ids_input, device=patch_tokens.device)
                patch_tokens_chunk = patch_tokens[frame_ids_tensor]
                images_chunk = self.vgg_input[frame_ids_tensor.cpu()].to(self.device, non_blocking=True)

                self.run_single_temporal_chunk(
                    set_idx,
                    chunk_set,
                    patch_tokens_chunk,
                    images_chunk,
                    chunk_idx,
                    frame_ids,
                    reversed_input=reverse_input,
                )
            print()
            
            
    def run_single_loop_centric_chunks(self, patch_tokens_chunk, images_chunk, item):
        tokens_chunk, patch_start_idx = self.model.aggregator.forward_aggregator(patch_tokens_chunk, *images_chunk.shape[-2:])
        cam_tokens_chunk, patch_tokens_chunk = tokens_chunk[-1][:, :, 0:1, :], torch.stack(tokens_chunk)[:, :, :, 5:, :]
        prediction_chunk = self.model.run_heads(cam_tokens_chunk, patch_tokens_chunk, images_chunk, patch_start_idx)
        
        extrinsic, intrinsic = pose_encoding_to_extri_intri(prediction_chunk['pose_enc'], images_chunk.shape[-2:])
        depth_map, depth_conf = prediction_chunk['depth'], prediction_chunk['depth_conf']
        del prediction_chunk
        
        extrinsic = extrinsic.squeeze(0).cpu().float().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().float().numpy()
        depth_map = depth_map.squeeze(0).cpu().float().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().float().numpy()
        
        intrinsic, depth_map = align_intrinsics_and_depth(intrinsic, self.intrinsic_ref, depth_map[..., 0])
        points_3d = unproject_depth_map_to_point_map(depth_map[..., None], extrinsic, intrinsic)
        
        prediction_chunk = {'extrinsic': extrinsic, 'intrinsic': intrinsic, 'depth_map': depth_map, 
                            'depth_conf': depth_conf, 'points_3d': points_3d}
        
        self.loop_prediction_list.append((item, prediction_chunk))
        
        
    def run_loop_centric_chunks(self, patch_tokens, loop_results):
        print("Process Loop-Centric Chunks")
        for idx, item in enumerate(loop_results):
            start1, end1 = item[1]
            start2, end2 = item[3]
            print(f"\tProcessing chunk {idx + 1:3d}/{len(loop_results)}: images {start1:4d} to {end1 - 1:4d} & {start2:4d} to {end2 - 1:4d}", end='\r', flush=True)
            
            patch_tokens_chunk = torch.cat([patch_tokens[start1: end1], patch_tokens[start2: end2]], dim=0)
            images_chunk = torch.cat([self.vgg_input[start1: end1], self.vgg_input[start2: end2]], dim=0).to(self.device, non_blocking=True)
            
            self.run_single_loop_centric_chunks(patch_tokens_chunk, images_chunk, item)
        print('\n')


    def _build_frame_to_chunk_map(self, chunk_set):
        frame_to_chunk = {}
        for chunk_idx, chunk_info in enumerate(chunk_set["temporal_chunk_list"]):
            frame_ids = chunk_info["frame_ids"]
            if not frame_ids:
                continue
            center = (len(frame_ids) - 1) * 0.5
            for local_idx, frame_id in enumerate(frame_ids):
                dist = abs(local_idx - center)
                prev = frame_to_chunk.get(frame_id)
                if prev is None or dist < prev[2]:
                    frame_to_chunk[frame_id] = (chunk_idx, local_idx, dist)

        chunk_set["frame_to_chunk"] = {
            frame_id: (chunk_idx, local_idx)
            for frame_id, (chunk_idx, local_idx, _) in frame_to_chunk.items()
        }


    def build_frame_to_chunk_maps(self):
        for chunk_set in self.chunk_sets:
            self._build_frame_to_chunk_map(chunk_set)


    def _compute_sim3_from_shared_frames(self, pred_fine, pred_coarse, frame_pairs):
        depth_thresh = self.args.depth_diff_threshold
        min_points = self.args.skip_edge_min_points
        depth_relax = self.args.skip_edge_depth_relax

        def _collect(thresh):
            points_f_list = []
            points_c_list = []
            total = 0

            for fine_idx, coarse_idx in frame_pairs:
                depth_f = pred_fine['depth_map'][fine_idx]
                depth_c = pred_coarse['depth_map'][coarse_idx]
                conf_f = pred_fine['depth_conf'][fine_idx]
                conf_c = pred_coarse['depth_conf'][coarse_idx]

                conf_f_mean = conf_f.mean()
                conf_c_mean = conf_c.mean()

                mask = (np.abs(depth_f - depth_c) < thresh) \
                    * (conf_f > conf_f_mean * self.args.conf_threshold) \
                    * (conf_c > conf_c_mean * self.args.conf_threshold)

                if mask.sum() == 0:
                    continue

                points_f = pred_fine['points_3d'][fine_idx][mask]
                points_c = pred_coarse['points_3d'][coarse_idx][mask]

                if points_f.size == 0 or points_c.size == 0:
                    continue

                points_f_list.append(points_f)
                points_c_list.append(points_c)
                total += mask.sum()

            return points_f_list, points_c_list, int(total)

        points_f_list, points_c_list, total = _collect(depth_thresh)
        if total < min_points:
            points_f_list, points_c_list, total = _collect(depth_thresh + depth_relax)

        if total < min_points or not points_f_list:
            return None, 0

        points_f = np.concatenate(points_f_list, axis=0)
        points_c = np.concatenate(points_c_list, axis=0)

        s, R, t = umeyama_sim3(points_f, points_c)
        return (s, R, t), total


    def _compose_sim3(self, sim3_src, sim3_dst):
        s_a, R_a, t_a = sim3_src
        s_b, R_b, t_b = sim3_dst
        S_a = self.loop_optimizer.numpy_to_pypose_sim3(s_a, R_a, t_a)
        S_b = self.loop_optimizer.numpy_to_pypose_sim3(s_b, R_b, t_b)
        S_out = S_a.Inv() @ S_b
        return self.loop_optimizer.pypose_sim3_to_numpy(S_out)


    def _scaled_edge_weight(self, base_weight, score):
        """Skip-edge weight, scaled down when the alignment has few correspondences."""
        min_points = self.args.skip_edge_min_points
        return base_weight * min(1.0, score / max(min_points, 1))


    def _compute_loop_density_array(self):
        """Per-frame loop density: count of loop-match endpoints within W frames on each side, normalized to [0, 1]."""
        n_frames = len(self.image_paths)
        W = int(self.args.loop_density_window)
        density = np.zeros(n_frames, dtype=np.float32)
        pairs = getattr(self, "loop_frame_pairs", None) or []
        for idx1, idx2 in pairs:
            for c in (int(idx1), int(idx2)):
                lo = max(0, c - W)
                hi = min(n_frames, c + W + 1)
                density[lo:hi] += 1.0
        if density.max() > 0:
            density = density / density.max()
        return density

    def build_skip_edges(self):
        """Skip edges between non-adjacent fine chunks i and j, composed through a coarse chunk
        that shares frames with both: S_ij = S_{i->c}^{-1} S_{j->c}."""
        self.skip_edges = []
        if len(self.chunk_sets) < 2:
            return

        self.build_frame_to_chunk_maps()
        base_state = self.chunk_sets[0]
        base_map = base_state["frame_to_chunk"]

        edge_candidates = {}
        min_gap = self.args.skip_edge_min_gap
        base_weight = self.args.skip_edge_weight

        if base_weight <= 0:
            return

        for set_idx in range(1, len(self.chunk_sets)):
            chunk_set = self.chunk_sets[set_idx]

            for c_local_idx, chunk_info in enumerate(chunk_set["temporal_chunk_list"]):
                frame_ids = chunk_info["frame_ids"]
                if not frame_ids:
                    continue

                fine_groups = {}
                for coarse_local_idx, frame_id in enumerate(frame_ids):
                    fine_entry = base_map.get(frame_id)
                    if fine_entry is None:
                        continue

                    fine_chunk_idx, fine_local_idx = fine_entry
                    fine_groups.setdefault(fine_chunk_idx, []).append((fine_local_idx, coarse_local_idx))

                if len(fine_groups) < 2:
                    continue

                coarse_pred = chunk_info["prediction"]
                fine_to_coarse = {}

                for fine_chunk_idx, frame_pairs in fine_groups.items():
                    fine_pred = base_state["temporal_chunk_list"][fine_chunk_idx]["prediction"]
                    sim3, score = self._compute_sim3_from_shared_frames(fine_pred, coarse_pred, frame_pairs)
                    if sim3 is None:
                        continue
                    fine_to_coarse[fine_chunk_idx] = {"sim3": sim3, "score": score}

                if len(fine_to_coarse) < 2:
                    continue

                fine_indices = sorted(fine_to_coarse.keys())
                fine_min = fine_indices[0]
                fine_max = fine_indices[-1]

                if fine_max - fine_min < min_gap:
                    continue

                sim3_min = fine_to_coarse[fine_min]["sim3"]
                sim3_max = fine_to_coarse[fine_max]["sim3"]
                skip_sim3 = self._compose_sim3(sim3_min, sim3_max)

                score = min(fine_to_coarse[fine_min]["score"], fine_to_coarse[fine_max]["score"])
                weight = self._scaled_edge_weight(base_weight, score)
                key = (fine_max, fine_min)

                if key not in edge_candidates or score > edge_candidates[key]["score"]:
                    edge_candidates[key] = {
                        "sim3": skip_sim3,
                        "score": score,
                        "weight": weight,
                    }

        for (chunk_i, chunk_j), data in edge_candidates.items():
            self.skip_edges.append((chunk_i, chunk_j, data["sim3"], data["weight"]))


    def _free_coarse_predictions(self):
        """Drop coarse-chunk point/depth arrays once the skip edges are built.
        Coarse predictions only feed the fine-coarse Sim(3) used to form skip edges.
        The final point cloud is assembled from the fine chunks alone."""
        freed = 0
        for chunk_set in self.chunk_sets[1:]:
            for chunk_info in chunk_set.get("temporal_chunk_list", []):
                pred = chunk_info.get("prediction")
                if not isinstance(pred, dict):
                    continue
                for k in ("points_3d", "depth_map", "depth_conf"):
                    v = pred.pop(k, None)
                    if v is not None:
                        try: freed += v.nbytes
                        except Exception: pass
                        del v
        if freed:
            import gc; gc.collect()
            print(f"Freed coarse chunk predictions: {freed/2**20:.0f} MiB")


    def build_pose_graph_edges(self):
        edges = []
        base_state = self.chunk_sets[0]

        # Fine sequential
        for idx, sim3 in enumerate(base_state["sim3_list"], start=1):
            edges.append((idx, idx - 1, sim3, 1.0))

        for chunk_idx_i, chunk_idx_j, sim3 in self.loop_sim3_list:
            edges.append((chunk_idx_i, chunk_idx_j, sim3, 1.0))

        for edge in self.skip_edges:
            edges.append(edge)

        return edges


    def _umeyama_align_overlapping_predictions(
        self,
        pred_a,
        pred_b,
        local_pairs,
        depth_multiplier=2.0,
    ):
        """
        Align two VGGT prediction dicts on paired local frame indices.
        Uses depth agreement + per-frame mean confidence (same as fine loop / temporal overlap).
        """
        depth_thresh = self.args.depth_diff_threshold * depth_multiplier

        def _collect(thresh):
            points_a_list = []
            points_b_list = []
            total = 0

            for idx_a, idx_b in local_pairs:
                depth_a = pred_a["depth_map"][idx_a]
                depth_b = pred_b["depth_map"][idx_b]
                conf_a = pred_a["depth_conf"][idx_a]
                conf_b = pred_b["depth_conf"][idx_b]

                mask = (np.abs(depth_a - depth_b) < thresh) \
                    * (conf_a > conf_a.mean() * self.args.conf_threshold) \
                    * (conf_b > conf_b.mean() * self.args.conf_threshold)

                if mask.sum() == 0:
                    continue

                points_a_list.append(pred_a["points_3d"][idx_a][mask])
                points_b_list.append(pred_b["points_3d"][idx_b][mask])
                total += int(mask.sum())

            if not points_a_list:
                return None

            return (
                np.concatenate(points_a_list, axis=0),
                np.concatenate(points_b_list, axis=0),
                total,
            )

        collected = _collect(depth_thresh)

        if collected is None:
            return None, 0

        points_a, points_b, total = collected
        s, R, t = umeyama_sim3(points_a, points_b)
        return (s, R, t), total


    def get_sim3_for_single_loop(self, chunk_idx, chunk_range, predictions, target):
        if target == False:
            chunk_begin = chunk_range[0] - self.chunk_list[chunk_idx][0]
            chunk_end = chunk_begin + chunk_range[1] - chunk_range[0]
            loop_local_pairs = [(i, i) for i in range(chunk_end - chunk_begin)]
        else:
            n_frames = chunk_range[1] - chunk_range[0]
            chunk_begin = chunk_range[0] - self.chunk_list[chunk_idx][0]
            chunk_end = chunk_begin + n_frames
            loop_start = predictions["points_3d"].shape[0] - n_frames
            loop_local_pairs = [(loop_start + i, i) for i in range(n_frames)]

        chunk_data = self.temporal_chunk_list[chunk_idx]["prediction"]
        loop_pred = {
            "depth_map": predictions["depth_map"],
            "depth_conf": predictions["depth_conf"],
            "points_3d": predictions["points_3d"],
        }
        temporal_pred = {
            "depth_map": chunk_data["depth_map"][chunk_begin:chunk_end],
            "depth_conf": chunk_data["depth_conf"][chunk_begin:chunk_end],
            "points_3d": chunk_data["points_3d"][chunk_begin:chunk_end],
        }

        result, _ = self._umeyama_align_overlapping_predictions(
            loop_pred, temporal_pred, loop_local_pairs
        )
        if result is None:
            raise RuntimeError(
                f"Loop-temporal alignment produced no correspondences "
                f"(chunk {chunk_idx}, range {chunk_range})"
            )
        s, R, t = result
        return s, R, t
    
    
    def get_sim3_for_loops(self):
        print("Estimate Sim(3) for Loop")
        for idx, item in enumerate(self.loop_prediction_list):
            print(f"\tSim(3) Align {idx + 1:3d}/{len(self.loop_prediction_list)}", end='\r', flush=True)
            chunk_idx_i, chunk_range_i = item[0][0], item[0][1]
            chunk_idx_j, chunk_range_j = item[0][2], item[0][3]
            predictions = item[1]
            
            s_i, R_i, t_i = self.get_sim3_for_single_loop(chunk_idx_i, chunk_range_i, predictions, target=False)
            s_j, R_j, t_j = self.get_sim3_for_single_loop(chunk_idx_j, chunk_range_j, predictions, target=True)
            
            s_ij, R_ij, t_ij = compute_sim3_ij((s_i, R_i, t_i), (s_j, R_j, t_j))
            
            self.loop_sim3_list.append((chunk_idx_i, chunk_idx_j, (s_ij, R_ij, t_ij)))
        print('\n')

    def align_chunks(self, sim3_list):
        print("Align Temporal Chunks")

        frame_points = {}
        frame_confs  = {}

        for idx, (start, end) in enumerate(self.chunk_list):
            chunk_data = self.temporal_chunk_list[idx]['prediction']

            if idx != 0:
                print(f"\tAligning {idx:3d} -> {idx - 1:3d} (Total {self.num_chunks - 1})", end="\r", flush=True)
                s, R, t = sim3_list[idx - 1]
                chunk_data["points_3d"] = apply_sim3_direct(chunk_data["points_3d"], s, R, t)

            pts_3d_chunk = chunk_data["points_3d"]
            conf_chunk   = chunk_data["depth_conf"]

            T_local = pts_3d_chunk.shape[0]

            for local_f in range(T_local):
                global_f = start + local_f

                pts_f  = pts_3d_chunk[local_f]
                conf_f = conf_chunk[local_f]

                if global_f not in frame_points:
                    frame_points[global_f] = []
                    frame_confs[global_f]  = []

                frame_points[global_f].append(pts_f)
                frame_confs[global_f].append(conf_f)

        print()

        points_list = []
        colors_list = []

        for f in sorted(frame_points.keys()):
            pts_list  = frame_points[f]
            conf_list = frame_confs[f]

            pts_stack  = np.stack(pts_list, axis=0)
            conf_stack = np.stack(conf_list, axis=0)

            pts_avg  = pts_stack.mean(axis=0)
            conf_avg = conf_stack.mean(axis=0)

            pts_flat   = pts_avg.reshape(-1, 3)
            conf_flat  = conf_avg.reshape(-1)

            img = (self.vgg_input[f].cpu().detach().numpy().transpose(1, 2, 0))
            colors_flat = (img.reshape(-1, 3) * 255).astype(np.uint8)

            conf_threshold = conf_flat.mean() * self.args.point_conf_threshold

            pts_out, colors_out = confident_pointcloud(
                points=pts_flat,
                colors=colors_flat,
                confs=conf_flat,
                conf_threshold=conf_threshold,
                sample_ratio=self.args.sampling_ratio,
            )

            points_list.append(pts_out)
            colors_list.append(colors_out)

        points_3d = np.concatenate(points_list, axis=0)
        colors    = np.concatenate(colors_list, axis=0)

        return points_3d, colors

    
    @torch.no_grad()
    def run(self):
        start = time.time()

        patch_tokens, loop_results = self.run_loop_detection(self.args)

        self.run_temporal_chunks(patch_tokens)
        self.run_loop_centric_chunks(patch_tokens, loop_results)
        self.get_sim3_for_loops()
        self.build_skip_edges()
        self._free_coarse_predictions()

        edges = self.build_pose_graph_edges()
        initial_poses = self.loop_optimizer.sequential_to_absolute_poses(self.chunk_sets[0]["sim3_list"])
        optimized_abs = self.loop_optimizer.optimize_graph(initial_poses, edges)

        sim3_list = self.loop_optimizer.absolute_to_sequential_transforms(optimized_abs)

        sim3_list = accumulate_sim3_transforms(sim3_list)

        points_3d, colors = self.align_chunks(sim3_list)

        end = time.time()
        print("Elapsed Time for Whole Process: ", end - start)

        save_camera_poses(self.camera_extrinsics, self.camera_intrinsics, sim3_list, self.args.output_path, num_imgs=self.vgg_input.shape[0])
        if self.args.save_points:
            trimesh.PointCloud(points_3d, colors=colors).export(os.path.join(self.args.output_path, "./points_3d.ply"))


import numpy as np
import torch
import pypose as pp
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple

cpp_version = False
try:
    import sim3solve
    cpp_version = True
except Exception as e:
    print(f"Sim3solve of C++ Version failed, Will using Python Version.")

from fastloop.solve_python import solve_system_py

import time

class Sim3LoopOptimizer:
    """
    Loop closure optimizer for sequences of Sim3 transformations
    
    Input:
    - sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]]
      Each element is (s, R, t), where s is scalar scale, R is [3,3] rotation matrix, t is [3,] translation vector
    - loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]]
      Each element is (i, j, (s, R, t)), representing a loop closure constraint from frame i to frame j
    
    Output:
    - Optimized sequential_transforms
    """
    
    def __init__(self, args, device='cpu'):
        self.device = device
        self.args = args
        self.solve_system_version = 'cpp'

        if not cpp_version:
            self.solve_system_version = 'python'
    
    def numpy_to_pypose_sim3(self, s: float, R_mat: np.ndarray, t_vec: np.ndarray) -> pp.Sim3:
        """Convert numpy s,R,t to pypose Sim3"""
        q = R.from_matrix(R_mat).as_quat()  # [x,y,z,w]
        # pypose requires [t, q, s] format
        data = np.concatenate([t_vec, q, np.array([s])])
        return pp.Sim3(torch.from_numpy(data).float().to(self.device))
    
    def pypose_sim3_to_numpy(self, sim3: pp.Sim3) -> Tuple[float, np.ndarray, np.ndarray]:
        """Convert pypose Sim3 to numpy s,R,t"""
        data = sim3.data.cpu().numpy()
        t = data[:3]
        q = data[3:7]  # [x,y,z,w]
        s = data[7]
        R_mat = R.from_quat(q).as_matrix()
        return s, R_mat, t
    
    def sequential_to_absolute_poses(self, sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]]) -> torch.Tensor:
        """
        Convert sequential relative transforms to absolute pose sequence
        S_01, S_12, S_23, ... -> T_0, T_1, T_2, T_3, ...
        Where T_i is the transform from world coordinate to frame i
        """
        n = len(sequential_transforms) + 1
        poses = []

        identity = pp.Sim3(torch.tensor([0., 0., 0., 0., 0., 0., 1., 1.], device=self.device))
        poses.append(identity)
        
        current_pose = identity
        for s, R_mat, t_vec in sequential_transforms:
            rel_transform = self.numpy_to_pypose_sim3(s, R_mat, t_vec)
            current_pose = current_pose @ rel_transform
            poses.append(current_pose)
        
        return torch.stack(poses)
    
    def absolute_to_sequential_transforms(self, absolute_poses: pp.Sim3) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Convert absolute pose sequence back to sequential relative transforms
        T_0, T_1, T_2, ... -> S_01, S_12, S_23, ...
        """
        sequential_transforms = []
        n = absolute_poses.shape[0]
        
        for i in range(n - 1):
            rel_transform = absolute_poses[i].Inv() @ absolute_poses[i + 1]
            s, R_mat, t_vec = self.pypose_sim3_to_numpy(rel_transform)
            sequential_transforms.append((s, R_mat, t_vec))
        
        return sequential_transforms
    
    def SE3_to_Sim3(self, x: torch.Tensor) -> pp.Sim3:
        """Convert SE3 to Sim3 (add unit scale)"""
        ones = torch.ones_like(x[..., :1])
        out = torch.cat((x, ones), dim=-1)
        return pp.Sim3(out)
    
    def build_loop_constraints(self, 
                             loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build loop closure constraints"""
        if not loop_constraints:
            return torch.empty(0, 8, device=self.device), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
        loop_transforms = []
        ii_loop = []
        jj_loop = []
        
        for i, j, (s, R_mat, t_vec) in loop_constraints:
            loop_sim3 = self.numpy_to_pypose_sim3(s, R_mat, t_vec)
            loop_transforms.append(loop_sim3.data)
            ii_loop.append(i)
            jj_loop.append(j)
        
        dSloop = pp.Sim3(torch.stack(loop_transforms))
        ii_loop = torch.tensor(ii_loop, dtype=torch.long, device=self.device)
        jj_loop = torch.tensor(jj_loop, dtype=torch.long, device=self.device)
        
        return dSloop, ii_loop, jj_loop

    def build_edge_constraints(self, edges: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray], float]]):
        """Build general Sim3 edge constraints with optional weights."""
        if not edges:
            return (
                torch.empty(0, 8, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, device=self.device),
            )

        transforms = []
        ii = []
        jj = []
        weights = []

        for edge in edges:
            if len(edge) == 4:
                i, j, (s, R_mat, t_vec), w = edge
            else:
                i, j, (s, R_mat, t_vec) = edge
                w = 1.0

            edge_sim3 = self.numpy_to_pypose_sim3(s, R_mat, t_vec)
            transforms.append(edge_sim3.data)
            ii.append(i)
            jj.append(j)
            weights.append(float(w))

        dS = pp.Sim3(torch.stack(transforms))
        ii = torch.tensor(ii, dtype=torch.long, device=self.device)
        jj = torch.tensor(jj, dtype=torch.long, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return dS, ii, jj, weights
    
    def residual(self, Ginv, input_poses, dSloop, ii, jj, jacobian=False):
        """Compute residuals (modified from original code)"""
        def _residual(C, Gi, Gj):
            out = C @ pp.Exp(Gi) @ pp.Exp(Gj).Inv()
            return out.Log().tensor()
        
        pred_inv_poses = pp.Sim3(input_poses).Inv()
        
        n, _ = pred_inv_poses.shape
        if n > 1:
            kk = torch.arange(1, n, device=self.device)
            ll = kk - 1
            Ti = pred_inv_poses[kk]
            Tj = pred_inv_poses[ll]
            dSij = Tj @ Ti.Inv()
        else:
            kk = torch.empty(0, dtype=torch.long, device=self.device)
            ll = torch.empty(0, dtype=torch.long, device=self.device)
            dSij = pp.Sim3(torch.empty(0, 8, device=self.device))
        
        constants = torch.cat((dSij.data, dSloop.data), dim=0) if dSloop.shape[0] > 0 else dSij.data
        if constants.shape[0] > 0:
            constants = pp.Sim3(constants)
            iii = torch.cat((kk, ii))
            jjj = torch.cat((ll, jj))
            resid = _residual(constants, Ginv[iii], Ginv[jjj])
        else:
            iii = torch.empty(0, dtype=torch.long, device=self.device)
            jjj = torch.empty(0, dtype=torch.long, device=self.device)
            resid = torch.empty(0, device=self.device)
        
        if not jacobian:
            return resid
        
        if constants.shape[0] > 0:
            def batch_jacobian(func, x):
                def _func_sum(*x):
                    return func(*x).sum(dim=0)
                _, b, c = torch.autograd.functional.jacobian(_func_sum, x, vectorize=True)
                from einops import rearrange
                return rearrange(torch.stack((b, c)), 'N O B I -> N B O I', N=2)
            
            J_Ginv_i, J_Ginv_j = batch_jacobian(_residual, (constants, Ginv[iii], Ginv[jjj]))
        else:
            J_Ginv_i = torch.empty(0, device=self.device)
            J_Ginv_j = torch.empty(0, device=self.device)
        
        return resid, (J_Ginv_i, J_Ginv_j, iii, jjj)

    def residual_graph(self, Ginv, dS, ii, jj, weights=None, jacobian=False):
        """Compute residuals for arbitrary Sim3 edges."""
        def _residual(C, Gi, Gj):
            out = C @ pp.Exp(Gi) @ pp.Exp(Gj).Inv()
            return out.Log().tensor()

        if dS.shape[0] == 0:
            if not jacobian:
                return torch.empty(0, device=self.device)
            empty = torch.empty(0, device=self.device)
            return empty, (empty, empty, ii, jj)

        resid = _residual(dS, Ginv[ii], Ginv[jj])

        if weights is not None and weights.numel() > 0:
            w = torch.sqrt(weights).unsqueeze(-1)
            resid = resid * w

        if not jacobian:
            return resid

        def batch_jacobian(func, x):
            def _func_sum(*x):
                return func(*x).sum(dim=0)
            _, b, c = torch.autograd.functional.jacobian(_func_sum, x, vectorize=True)
            from einops import rearrange
            return rearrange(torch.stack((b, c)), 'N O B I -> N B O I', N=2)

        J_Ginv_i, J_Ginv_j = batch_jacobian(_residual, (dS, Ginv[ii], Ginv[jj]))

        if weights is not None and weights.numel() > 0:
            w = torch.sqrt(weights).view(-1, 1, 1)
            J_Ginv_i = J_Ginv_i * w
            J_Ginv_j = J_Ginv_j * w

        return resid, (J_Ginv_i, J_Ginv_j, ii, jj)

    def _robust_weights(self, resid, kernel, delta):
        if kernel is None or kernel == "none":
            return torch.ones(resid.shape[0], device=resid.device)

        eps = 1e-12
        delta = float(delta) if delta is not None and delta > 0 else 1.0
        r = torch.linalg.norm(resid, dim=1).clamp_min(eps)

        if kernel == "huber":
            w = torch.where(r <= delta, torch.ones_like(r), delta / r)
        elif kernel == "cauchy":
            w = 1.0 / (1.0 + (r / delta) ** 2)
        else:
            w = torch.ones_like(r)

        return w

    def optimize_graph(
        self,
        initial_poses: torch.Tensor,
        edges: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray], float]],
        max_iterations: int = None,
        lambda_init: float = None,
    ) -> pp.Sim3:
        """Optimize a pose graph with arbitrary Sim3 edges."""
        print("Pose Graph Optimization")

        if max_iterations is None:
            max_iterations = self.args.loop_max_iteration
        if lambda_init is None:
            lambda_init = self.args.loop_lambda_init

        if not edges:
            print("Warning: No edges provided, returning initial poses")
            return pp.Sim3(initial_poses)

        input_poses = initial_poses.to(self.device)
        dS, ii, jj, weights = self.build_edge_constraints(edges)

        Ginv = pp.Sim3(input_poses).Inv().Log()
        lmbda = lambda_init
        residual_history = []
        kernel = getattr(self.args, "robust_kernel", "none")
        delta = getattr(self.args, "robust_delta", 1.0)

        cost_cnt = 0
        for itr in range(max_iterations):
            resid_raw, (J_Ginv_i, J_Ginv_j, iii, jjj) = self.residual_graph(
                Ginv, dS, ii, jj, weights=None, jacobian=True
            )

            if resid_raw.numel() == 0:
                print("No residuals to optimize")
                break

            base_weights = weights if weights.numel() > 0 else torch.ones(resid_raw.shape[0], device=self.device)
            robust_weights = self._robust_weights(resid_raw, kernel, delta)
            combined_weights = base_weights * robust_weights
            sqrt_w = torch.sqrt(combined_weights).unsqueeze(-1)

            resid = resid_raw * sqrt_w
            J_Ginv_i = J_Ginv_i * sqrt_w.view(-1, 1, 1)
            J_Ginv_j = J_Ginv_j * sqrt_w.view(-1, 1, 1)

            current_cost = resid.square().mean().item()
            residual_history.append(current_cost)

            try:
                begin_time = time.time()
                if self.solve_system_version == 'cpp':
                    delta_pose, = sim3solve.solve_system(
                        J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1
                    )
                elif self.solve_system_version == 'python':
                    delta_pose = solve_system_py(
                        J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1
                    )
                else:
                    print("Solver version has not been chosen! ('python' or 'cpp')")
                end_time = time.time()
            except Exception as e:
                print(f"Solver failed at iteration {itr}: {e}")
                break

            Ginv_tmp = Ginv + delta_pose
            new_resid_raw = self.residual_graph(Ginv_tmp, dS, ii, jj, weights=None)
            if new_resid_raw.numel() > 0:
                new_robust_weights = self._robust_weights(new_resid_raw, kernel, delta)
                new_combined = base_weights * new_robust_weights
                new_resid = new_resid_raw * torch.sqrt(new_combined).unsqueeze(-1)
                new_cost = new_resid.square().mean().item()
            else:
                new_cost = float('inf')

            if new_cost < current_cost:
                Ginv = Ginv_tmp
                lmbda /= 2
            else:
                lmbda *= 2

            print(
                f"\tIteration {itr}: cost {current_cost:.14f} -> {new_cost:.14f} "
                f"({'accepted' if new_cost < current_cost else 'rej'}) | "
                f"Time {(end_time - begin_time) * 1000:.4f} ms",
                end='\r',
                flush=True,
            )

            if np.abs(current_cost - new_cost) < 1e-8:
                cost_cnt += 1
                if cost_cnt == 20:
                    break

            if (current_cost < 1e-5) and (itr >= 4):
                if len(residual_history) >= 5:
                    improvement_ratio = residual_history[-5] / residual_history[-1]
                    if improvement_ratio < 1.5:
                        print(f"Converged at iteration {itr}")
                        break

        print('\n')

        optimized_absolute_poses = pp.Exp(Ginv).Inv()
        return optimized_absolute_poses
    
    def optimize(self, 
                sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]],
                loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]],
                max_iterations: int = None,
                lambda_init: float = None) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Main optimization function
        
        Args:
            sequential_transforms: Input sequence of transforms
            loop_constraints: List of loop closure constraints
            max_iterations: Maximum iterations
            lambda_init: Initial lambda for L-M algorithm
        
        Returns:
            Optimized sequence of transforms
        """
        print("Loop Optimization")
        if max_iterations is None:
            max_iterations = self.args.loop_max_iteration
        if lambda_init is None:
            lambda_init = self.args.loop_lambda_init

        input_poses = self.sequential_to_absolute_poses(sequential_transforms)
        
        dSloop, ii_loop, jj_loop = self.build_loop_constraints(loop_constraints)
        
        if len(loop_constraints) == 0:
            print("Warning: No loop constraints provided, returning original transforms")
            return sequential_transforms
        
        Ginv = pp.Sim3(input_poses).Inv().Log()
        lmbda = lambda_init
        residual_history = []
        
        # print(f"Starting optimization with {len(sequential_transforms)} poses and {len(loop_constraints)} loop constraints")
        
        # L-M loop
        cost_cnt = 0
        for itr in range(max_iterations):
            resid, (J_Ginv_i, J_Ginv_j, iii, jjj) = self.residual(
                Ginv, input_poses, dSloop, ii_loop, jj_loop, jacobian=True)
            
            if resid.numel() == 0:
                print("No residuals to optimize")
                break
                
            current_cost = resid.square().mean().item()
            residual_history.append(current_cost)
            
            try: # Solve linear system
                begin_time = time.time()
                if self.solve_system_version == 'cpp':
                    delta_pose, = sim3solve.solve_system(
                        J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1)
                elif self.solve_system_version == 'python':
                    delta_pose = solve_system_py(
                        J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1)
                else:
                    print(f"Solver version has not been chosen! ('python' or 'cpp')")
                end_time = time.time()
            except Exception as e:
                print(f"Solver failed at iteration {itr}: {e}")
                break
            
            Ginv_tmp = Ginv + delta_pose
            
            new_resid = self.residual(Ginv_tmp, input_poses, dSloop, ii_loop, jj_loop)
            new_cost = new_resid.square().mean().item() if new_resid.numel() > 0 else float('inf')
            
            # L-M
            if new_cost < current_cost:
                Ginv = Ginv_tmp
                lmbda /= 2
            else:
                lmbda *= 2
            print(f"\tIteration {itr}: cost {current_cost:.14f} -> {new_cost:.14f} "
                  f"({'accepted' if new_cost < current_cost else 'rej'}) | "
                  f"Time {(end_time - begin_time)*1000:.4f} ms", end='\r', flush=True)

            if np.abs(current_cost - new_cost) < 1e-8:
                cost_cnt += 1
                if cost_cnt == 20:
                    break

            if (current_cost < 1e-5) and (itr >= 4):
                if len(residual_history) >= 5:
                    improvement_ratio = residual_history[-5] / residual_history[-1]
                    if improvement_ratio < 1.5:
                        print(f"Converged at iteration {itr}")
                        break
        
        print('\n')
        
        optimized_absolute_poses = pp.Exp(Ginv).Inv()
        
        optimized_sequential = self.absolute_to_sequential_transforms(optimized_absolute_poses)
        
        # print(f"Optimization completed. Final cost: {residual_history[-1] if residual_history else 'N/A'}")
        
        return optimized_sequential
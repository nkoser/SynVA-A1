import os
import pickle
from pathlib import Path

import numpy as np
import torch


def _resample_closed_ring(points: np.ndarray, num_points: int) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected ring points with shape [N, 3], got {points.shape}.")
    if len(points) < 3:
        raise ValueError("Need at least three points to resample a ring.")
    if num_points < 3:
        raise ValueError("num_points must be at least 3.")

    diffs = np.roll(points, -1, axis=0) - points
    seg_lengths = np.linalg.norm(diffs, axis=1)
    if np.all(seg_lengths < 1e-8):
        raise ValueError("Ring perimeter is degenerate.")

    cum_length = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_length = float(cum_length[-1])
    sample_positions = np.linspace(0.0, total_length, num=num_points, endpoint=False)
    resampled = np.zeros((num_points, 3), dtype=np.float32)

    for i, sample_pos in enumerate(sample_positions):
        seg_idx = min(np.searchsorted(cum_length, sample_pos, side='right') - 1, len(points) - 1)
        seg_start = points[seg_idx]
        seg_end = points[(seg_idx + 1) % len(points)]
        seg_length = seg_lengths[seg_idx]
        if seg_length < 1e-8:
            resampled[i] = seg_start
            continue
        alpha = (sample_pos - cum_length[seg_idx]) / seg_length
        resampled[i] = (1.0 - alpha) * seg_start + alpha * seg_end
    return resampled


def _align_ring_to_reference(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if points.shape != reference.shape:
        raise ValueError(f"points and reference must have the same shape, got {points.shape} vs {reference.shape}.")

    best = None
    best_error = None
    for candidate in (points, points[::-1].copy()):
        for shift in range(candidate.shape[0]):
            shifted = np.roll(candidate, -shift, axis=0)
            error = float(np.mean((shifted - reference) ** 2))
            if best_error is None or error < best_error:
                best = shifted.copy()
                best_error = error
    return best


class GHDDataset(torch.utils.data.Dataset):
    def __init__(self, ghd_chk_root, ghd_run, ghd_chk_name, ghd_reconstruct, cases, withscale=True, normalize=True):
        self.ghd_chk_root = ghd_chk_root
        self.ghd_run = ghd_run
        self.ghd_chk_name = ghd_chk_name
        self.ghd_reconstruct = ghd_reconstruct
        self.withscale = withscale
        self.cases = cases
        # get eigvec
        self.GHD_eigvec = self.ghd_reconstruct.canonical_ghd.GBH_eigvec
        # assemble dataset
        self.updated_cases, self.ghd, self.alignment, self.scale = self.assemble()
        # normalize
        self.normalize = normalize
        self.mean, self.std = self.normalize_()
        
    def assemble(self, print_jnl=False):
        updated_cases = []
        ghd = []
        alignment = []
        scale = []

        for case in self.cases:
            ghd_checkpoint = os.path.join(self.ghd_chk_root, case, self.ghd_run, self.ghd_chk_name)
            if os.path.exists(ghd_checkpoint):
                with open(ghd_checkpoint, 'rb') as f:
                    ghd_chk = pickle.load(f)
                ghd.append(ghd_chk['GHD_coefficient'].view(-1))
                R, s, T = ghd_chk['R'], ghd_chk['s'].abs(), ghd_chk['T']
                alignment.append(torch.cat((R.view(-1), s.view(-1), T.view(-1))).detach())
                scale.append(s.view(-1).detach())
                updated_cases.append(case)
            else:
                if print_jnl:
                    print("GHD checkpoint not found for case: ", case)
        print("{} cases out of {} have GHD checkpoint".format(len(updated_cases), len(self.cases)))
        return updated_cases, ghd, alignment, scale
    
    def normalize_(self):
        if self.withscale:
            dataset = torch.stack([torch.cat([ghd, scale]) for ghd, scale in zip(self.ghd, self.scale)], dim=0)
        else:
            dataset = torch.stack(self.ghd, dim=0)
        mean = dataset.mean(dim=0, keepdim=True)
        std = dataset.std(dim=0, keepdim=True) + 0.01
        return mean, std
    
    def __len__(self):
        return len(self.updated_cases)
    
    def __getitem__(self, idx):
        # case = self.updated_cases[idx]
        ghd = self.ghd[idx]
        scale = self.scale[idx]
        x = torch.cat([ghd, scale]) if self.withscale else ghd
        if self.normalize:
            x = (x - self.mean) / self.std
        return x.view(-1)
    
    def de_normalize(self, x):
        if not self.normalize:
            return x
        else:
            return x * self.std.to(x.device) + self.mean.to(x.device)
    
    def get_dim(self):
        x = self.__getitem__(0)
        if self.withscale:
            return x.shape[0] - 1
        else:
            return x.shape[0]
    
    def get_mean_std(self):
        if self.withscale:
            return self.mean[:, :-1], self.std[:, :-1]
        else:
            return self.mean, self.std
    
    def get_scale_mean_std(self):
        if self.withscale:
            return self.mean[:, -1], self.std[:, -1]
        else:
            return None, None
    
    def denorm_scale(self, scale):
        assert self.withscale
        return scale * self.std[:, -1].to(scale.device) + self.mean[:, -1].to(scale.device)


class OstiumGHDDataset(torch.utils.data.Dataset):
    def __init__(self,
                 ghd_chk_root,
                 alignment_root,
                 canonical_opa_chk_path,
                 cases,
                 ghd_run='vanilla',
                 ghd_chk_name='ghb_fitting_checkpoint.pkl',
                 withscale=True,
                 normalize=True,
                 ring_points=22):
        self.ghd_chk_root = ghd_chk_root
        self.alignment_root = alignment_root
        self.ghd_run = ghd_run
        self.ghd_chk_name = ghd_chk_name
        self.withscale = withscale
        self.normalize = normalize
        self.cases = cases

        self.ring_points = ring_points
        self.reference_ring, self.canonical_opening_idx = self.load_reference_ring(canonical_opa_chk_path)

        (
            self.updated_cases,
            self.ghd,
            self.scale,
            self.ostium_condition,
            self.alignment_rotation,
            self.alignment_translation,
        ) = self.assemble()
        self.target_mean, self.target_std, self.cond_mean, self.cond_std = self.compute_stats()

    def load_reference_ring(self, canonical_opa_chk_path):
        with open(canonical_opa_chk_path, 'rb') as f:
            canonical_chk = pickle.load(f)
        reference_ring = np.asarray(canonical_chk['op_v_coords'][0], dtype=np.float32)
        opening_idx = np.asarray(canonical_chk['op_v_indices'][0], dtype=np.int64)
        if self.ring_points is None:
            self.ring_points = min(22, len(opening_idx))
        if self.ring_points > len(opening_idx):
            raise ValueError(
                f"ring_points={self.ring_points} exceeds the canonical opening size ({len(opening_idx)})."
            )
        opening_selection = np.linspace(0, len(opening_idx), num=self.ring_points, endpoint=False, dtype=np.int64)
        opening_idx = torch.tensor(opening_idx[opening_selection], dtype=torch.long)
        reference_ring = _resample_closed_ring(reference_ring, self.ring_points)
        return reference_ring, opening_idx

    def load_case_ring(self, case):
        opa_checkpoint = os.path.join(self.alignment_root, case, 'opa_checkpoint.pkl')
        return self.load_ring_from_opa_checkpoint(opa_checkpoint)

    def load_ring_from_opa_checkpoint(self, opa_checkpoint):
        with open(opa_checkpoint, 'rb') as f:
            opa_chk = pickle.load(f)
        ring = np.asarray(opa_chk['op_v_coords'][0], dtype=np.float32)
        ring = _resample_closed_ring(ring, self.ring_points)
        ring = _align_ring_to_reference(ring, self.reference_ring)
        return torch.from_numpy(ring.reshape(-1)).float()

    def assemble(self, print_jnl=False):
        updated_cases = []
        ghd = []
        scale = []
        ostium_condition = []
        alignment_rotation = []
        alignment_translation = []

        for case in self.cases:
            case_root = Path(self.ghd_chk_root) / case
            ghd_checkpoint = case_root / self.ghd_run / self.ghd_chk_name
            opa_checkpoint = Path(self.alignment_root) / case / 'opa_checkpoint.pkl'
            if (not ghd_checkpoint.exists()) or (not opa_checkpoint.exists()):
                if print_jnl:
                    print(f"Missing checkpoint(s) for case: {case}")
                continue

            with open(ghd_checkpoint, 'rb') as f:
                ghd_chk = pickle.load(f)

            ghd.append(ghd_chk['GHD_coefficient'].view(-1).float())
            scale.append(ghd_chk['s'].abs().view(-1).float())
            ostium_condition.append(self.load_case_ring(case))
            alignment_rotation.append(ghd_chk['R'].view(-1).float())
            alignment_translation.append(ghd_chk['T'].view(-1).float())
            updated_cases.append(case)

        print(f"{len(updated_cases)} cases out of {len(self.cases)} have both pouch GHD and ostium checkpoints")
        return (
            updated_cases,
            ghd,
            scale,
            ostium_condition,
            alignment_rotation,
            alignment_translation,
        )

    def compute_stats(self):
        if self.withscale:
            targets = torch.stack([torch.cat([ghd, scale]) for ghd, scale in zip(self.ghd, self.scale)], dim=0)
        else:
            targets = torch.stack(self.ghd, dim=0)
        conditions = torch.stack(self.ostium_condition, dim=0)
        target_mean = targets.mean(dim=0, keepdim=True)
        target_std = targets.std(dim=0, keepdim=True, unbiased=False) + 0.01
        cond_mean = conditions.mean(dim=0, keepdim=True)
        cond_std = conditions.std(dim=0, keepdim=True, unbiased=False) + 0.01
        return target_mean, target_std, cond_mean, cond_std

    def __len__(self):
        return len(self.updated_cases)

    def __getitem__(self, idx):
        ghd = self.ghd[idx]
        scale = self.scale[idx]
        target = torch.cat([ghd, scale]) if self.withscale else ghd
        condition = self.ostium_condition[idx]
        if self.normalize:
            target = (target - self.target_mean.squeeze(0)) / self.target_std.squeeze(0)
            condition = (condition - self.cond_mean.squeeze(0)) / self.cond_std.squeeze(0)
        return {
            'target': target.view(-1),
            'condition': condition.view(-1),
            'case': self.updated_cases[idx],
            'alignment_rotation': self.alignment_rotation[idx].view(-1),
            'alignment_translation': self.alignment_translation[idx].view(-1),
        }

    def get_target_dim(self):
        target = self.__getitem__(0)['target']
        return target.shape[0]

    def get_ghd_dim(self):
        return self.ghd[0].shape[0]

    def get_cond_dim(self):
        return self.ostium_condition[0].shape[0]

    def get_mean_std(self):
        if self.withscale:
            return self.target_mean[:, :-1], self.target_std[:, :-1]
        return self.target_mean, self.target_std

    def get_scale_mean_std(self):
        if self.withscale:
            return self.target_mean[:, -1:], self.target_std[:, -1:]
        return None, None

    def denorm_scale(self, scale):
        if not self.withscale:
            return None
        return scale * self.target_std[:, -1:].to(scale.device) + self.target_mean[:, -1:].to(scale.device)

    def normalize_condition(self, condition):
        return (condition - self.cond_mean.to(condition.device)) / self.cond_std.to(condition.device)

    def denormalize_condition(self, condition):
        return condition * self.cond_std.to(condition.device) + self.cond_mean.to(condition.device)

    def get_condition_from_case(self, case, normalize=None):
        if normalize is None:
            normalize = self.normalize
        condition = self.load_case_ring(case)
        if normalize:
            condition = self.normalize_condition(condition.unsqueeze(0)).squeeze(0)
        return condition

    def get_condition_from_opa_checkpoint(self, opa_checkpoint, normalize=None):
        if normalize is None:
            normalize = self.normalize
        condition = self.load_ring_from_opa_checkpoint(opa_checkpoint)
        if normalize:
            condition = self.normalize_condition(condition.unsqueeze(0)).squeeze(0)
        return condition

    def get_canonical_opening_idx(self, device=None):
        if device is None:
            return self.canonical_opening_idx.clone()
        return self.canonical_opening_idx.to(device)
     

class CenterlineDataset(torch.utils.data.Dataset):
    def __init__(self, cl_chk_root: str, normalize=True, toss_threshold=0.005, device=torch.device('cuda:0')):
        self.cl_chk_root = cl_chk_root
        self.toss_threshold = toss_threshold
        self.cases, self.data, self.num_branch, self.num_fourier, self.fourier_per_branch = self.assemble()
        self.normalize = normalize
        self.norm_dict = self.normalize_()
        self.device = device

    def assemble(self, print_jnl=False):
        chk_files = [os.path.join(self.cl_chk_root, file) for file in os.listdir(self.cl_chk_root) if file.endswith('.pth')]
        cases = []
        start_end_vector = []
        split_centerline = []
        branch_length = []
        fouriers = []
        ghd = []
        scale = []
        relative_directions = []
        accurate_tangent = []

        toss_cases = []
        for chk_file in chk_files:
            chk = torch.load(chk_file)
            case = chk['label']
            if chk['fitting_loss']<self.toss_threshold:
                cases.append(chk['label'])
                # start_end_vector.append(chk['start_end_vector'])
                split_centerline.append(chk['split_centerline'])
                branch_length.append(torch.stack(chk['branch_length']).unsqueeze(0))  # [1, num_branch]
                fouriers.append(chk['fouriers'])
                ghd.append(chk['ghd'].unsqueeze(0))
                scale.append(chk['scale'].unsqueeze(0))
                relative_directions.append(chk['relative_directions'])
                accurate_tangent.append(torch.cat(chk['accurate_tangent']))
            else:
                toss_cases.append(chk['label'])
        temp = fouriers[0]
        num_branch, num_fourier, fourier_per_branch = temp.shape
        relative_directions = torch.cat([ten.view(1, 4*num_branch) for ten in relative_directions], dim=0)  # [1, 4*num_branch]
        ghd = torch.cat(ghd)
        fouriers = torch.cat([tensor_.view(1, num_branch*num_fourier*fourier_per_branch) for tensor_ in fouriers])
        branch_length = torch.cat(branch_length)
        print("{} cases have been loaded, {} cases have been tosses due to bad fitting".format(len(cases), len(toss_cases)))
        data = {'ghd': ghd, 'relative_directions': relative_directions, 'fouriers': fouriers, 'branch_length': branch_length,
                'accurate_tangent': accurate_tangent, 'scale': torch.cat(scale)}
        self.split_centerline = split_centerline
        return cases, data, num_branch, num_fourier, fourier_per_branch
    
    def normalize_(self):
        norm_dict = {}
        for key, value in self.data.items():
            if key != 'accurate_tangent':
                mean = value.mean(dim=0, keepdim=True)
                std = value.std(dim=0, keepdim=True) + 0.01
                norm_dict[key] = (mean, std)
        return norm_dict

    def de_normalize(self, data_dict):
        if not self.normalize:
            pass
        else:
            for key in [key_ for key_ in data_dict.keys() if key_ != 'accurate_tangent']:
                mean, std = self.norm_dict[key]
                data_dict[key] = data_dict[key] * std.to(data_dict[key].device) + mean.to(data_dict[key].device)
        return data_dict

    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        data_dict = {}  
        for key in self.data.keys():
            if key != 'accurate_tangent':
                mean, std = self.norm_dict[key]
                data_dict[key] = self.data[key][idx].view(1, -1)
                data_dict[key] = ((self.data[key][idx] - mean) / std if self.normalize else self.data[key][idx]).squeeze(0)
            else:
                data_dict[key] = self.data[key][idx]
            data_dict[key] = data_dict[key].to(self.device)
        return data_dict
    
    def return_norm_dict(self, device):
        norm_dict = {}
        for key in self.norm_dict.keys():
            mean, std = self.norm_dict[key]
            norm_dict[key] = (mean.to(device), std.to(device))
        return norm_dict
        
    





        

        





        

        

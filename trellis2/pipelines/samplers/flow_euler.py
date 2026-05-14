from typing import *
import csv
import os
import matplotlib
matplotlib.use("Agg")
import re
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
import torch.nn as nn
import torch.nn.functional as F
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """

    def __init__(
            self,
            sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    def _ensure_feature_estimator(self, model) -> None:
        self.estimator = model

    def _visualize_pca_feature_pointcloud(
            self,
            feature: torch.Tensor,
            n_components: int = 3,
            max_points: int = 12000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a 3D feature volume (B, C, R, R, R) to PCA-colored point cloud data.
        """
        if not isinstance(feature, torch.Tensor):
            raise TypeError(f"`feature` must be a torch.Tensor, but got {type(feature)}")
        if feature.dim() != 5:
            raise ValueError(f"`feature` must be 5D (B, C, R, R, R), but got shape {tuple(feature.shape)}")

        feature_np = feature.detach().cpu().float().numpy()
        batch_size, channels, depth, height, width = feature_np.shape
        if not (depth == height == width):
            raise ValueError(f"Expected cubic feature volume, but got shape {tuple(feature_np.shape)}")

        n_components = min(max(1, n_components), channels, 3)
        coords = np.stack(
            np.meshgrid(
                np.arange(depth, dtype=np.float32),
                np.arange(height, dtype=np.float32),
                np.arange(width, dtype=np.float32),
                indexing="ij"
            ),
            axis=-1
        ).reshape(-1, 3)
        coords_norm = coords / max(depth - 1, 1)
        features_2d = feature_np[0].reshape(channels, -1).T  # (R^3, C), 只可视化 batch 0
        feature_mean = np.mean(features_2d, axis=0, keepdims=True)
        feature_std = np.std(features_2d, axis=0, keepdims=True) + 1e-8
        features_scaled = (features_2d - feature_mean) / feature_std

        _, _, vh = np.linalg.svd(features_scaled, full_matrices=False)
        components = vh[:n_components].T
        pca_result = features_scaled @ components

        pca_normalized = np.zeros_like(pca_result, dtype=np.float32)
        for i in range(pca_result.shape[1]):
            pca_i = pca_result[:, i]
            pca_normalized[:, i] = (pca_i - pca_i.min()) / (pca_i.max() - pca_i.min() + 1e-8)

        rgb = np.zeros((pca_normalized.shape[0], 3), dtype=np.float32)
        rgb[:, :pca_normalized.shape[1]] = pca_normalized[:, :3]

        num_points = coords_norm.shape[0]
        if num_points > max_points:
            sample_idx = np.random.choice(num_points, size=max_points, replace=False)
        else:
            sample_idx = np.arange(num_points)
        return coords_norm[sample_idx], rgb[sample_idx]

    def _save_hook_feature_grid(
            self,
            features: Dict[str, torch.Tensor],
            step_tag: float,
            source_tag: str
    ) -> None:
        output_dir = os.path.join("outputs", "pca_feature_pointcloud")
        os.makedirs(output_dir, exist_ok=True)

        layer_order = ["norm-01", "norm-02", "final"]
        viewpoint_order = [(20, 35), (20, 125), (85, -90)]  # 3视图

        grouped: Dict[int, Dict[str, torch.Tensor]] = {}
        for key, value in features.items():
            match = re.match(r"block-(\d+)-(norm-01|norm-02|final)$", key)
            if match is None:
                continue
            block_idx = int(match.group(1))
            layer_name = match.group(2)
            grouped.setdefault(block_idx, {})[layer_name] = value

        if len(grouped) == 0:
            return

        block_ids = sorted(grouped.keys())
        n_rows = len(block_ids)
        n_cols = len(layer_order) * len(viewpoint_order)  # 3 layer x 3 view = 9
        fig = plt.figure(figsize=(n_cols * 2.1, n_rows * 2.2))

        for row_idx, block_idx in enumerate(block_ids):
            for layer_idx, layer_name in enumerate(layer_order):
                feat = grouped[block_idx].get(layer_name, None)
                coords_plot = None
                rgb_plot = None
                if feat is not None and feat.dim() == 3:
                    resolution = round(feat.shape[1] ** (1 / 3))
                    if resolution ** 3 == feat.shape[1]:
                        volume = feat.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[2], resolution, resolution, resolution)
                        coords_plot, rgb_plot = self._visualize_pca_feature_pointcloud(volume.detach())

                for view_idx, (elev, azim) in enumerate(viewpoint_order):
                    col_idx = layer_idx * len(viewpoint_order) + view_idx
                    subplot_idx = row_idx * n_cols + col_idx + 1
                    ax = fig.add_subplot(n_rows, n_cols, subplot_idx, projection="3d")
                    if coords_plot is not None and rgb_plot is not None:
                        ax.scatter(
                            coords_plot[:, 0], coords_plot[:, 1], coords_plot[:, 2],
                            c=rgb_plot, s=0.8, alpha=0.8, linewidths=0
                        )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_zlim(0, 1)
                    ax.view_init(elev=elev, azim=azim)
                    if row_idx == 0:
                        ax.set_title(f"{layer_name}\nview-{view_idx + 1}", fontsize=8)
                    if col_idx == 0:
                        ax.text2D(0.02, 0.5, f"block-{block_idx:02d}", transform=ax.transAxes, fontsize=8)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"step_{step_tag:07.4f}_{source_tag}.png")
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"\t * Saved PCA 3-view grid to {save_path}.")


    # def _visualize_pca_feature_pointcloud(
    #         self,
    #         feature: torch.Tensor,
    #         comment: Optional[str] = None,
    #         n_components: int = 10,
    #         max_points: int = 50000
    # ) -> None:
    #     """
    #     Visualize a 3D feature volume (B, C, R, R, R) with PCA color mapping and save files.
    #     """
    #     if not isinstance(feature, torch.Tensor):
    #         raise TypeError(f"`feature` must be a torch.Tensor, but got {type(feature)}")
    #     if feature.dim() != 5:
    #         raise ValueError(f"`feature` must be 5D (B, C, R, R, R), but got shape {tuple(feature.shape)}")
    #
    #     feature_np = feature.detach().cpu().float().numpy()
    #     batch_size, channels, depth, height, width = feature_np.shape
    #     if not (depth == height == width):
    #         raise ValueError(f"Expected cubic feature volume, but got shape {tuple(feature_np.shape)}")
    #
    #     output_dir = os.path.join("outputs", "pca_feature_pointcloud")
    #     os.makedirs(output_dir, exist_ok=True)
    #     log_path = os.path.join(output_dir, "pca_feature_log.csv")
    #
    #     n_components = min(max(1, n_components), channels, 10)
    #     coords = np.stack(
    #         np.meshgrid(
    #             np.arange(depth, dtype=np.float32),
    #             np.arange(height, dtype=np.float32),
    #             np.arange(width, dtype=np.float32),
    #             indexing="ij"
    #         ),
    #         axis=-1
    #     ).reshape(-1, 3)
    #     coords_norm = coords / max(depth - 1, 1)
    #
    #     def _save_pca_log(csv_path: str, explained_variance_ratio: np.ndarray,
    #                       cumulative_variance: float, row_comment: Optional[str]) -> None:
    #         ratios = list(explained_variance_ratio.astype(np.float64)) + [0.0] * (10 - len(explained_variance_ratio))
    #         fieldnames = ["Comment"] + [f"PC{i + 1}" for i in range(10)] + ["Total"]
    #         file_exists = os.path.exists(csv_path)
    #         with open(csv_path, "a", newline="") as csvfile:
    #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #             if not file_exists:
    #                 writer.writeheader()
    #             row_data = {
    #                 "Comment": row_comment if row_comment is not None else "",
    #                 **{f"PC{i + 1}": f"{ratios[i] * 100:.2f}%" for i in range(10)},
    #                 "Total": f"{cumulative_variance * 100:.2f}%"
    #             }
    #             writer.writerow(row_data)
    #
    #     for batch_idx in range(batch_size):
    #         features_2d = feature_np[batch_idx].reshape(channels, -1).T  # (R^3, C)
    #         feature_mean = np.mean(features_2d, axis=0, keepdims=True)
    #         feature_std = np.std(features_2d, axis=0, keepdims=True) + 1e-8
    #         features_scaled = (features_2d - feature_mean) / feature_std
    #
    #         # PCA via SVD
    #         _, singular_values, vh = np.linalg.svd(features_scaled, full_matrices=False)
    #         components = vh[:n_components].T
    #         pca_result = features_scaled @ components
    #         explained_variance = (singular_values ** 2) / max(features_scaled.shape[0] - 1, 1)
    #         explained_variance_ratio = explained_variance[:n_components] / np.sum(explained_variance + 1e-12)
    #         cumulative_variance = float(np.sum(explained_variance_ratio))
    #
    #         pca_normalized = np.zeros_like(pca_result, dtype=np.float32)
    #         for i in range(pca_result.shape[1]):
    #             pca_i = pca_result[:, i]
    #             pca_normalized[:, i] = (pca_i - pca_i.min()) / (pca_i.max() - pca_i.min() + 1e-8)
    #
    #         rgb = np.zeros((pca_normalized.shape[0], 3), dtype=np.float32)
    #         use_rgb = min(3, pca_normalized.shape[1])
    #         rgb[:, :use_rgb] = pca_normalized[:, :use_rgb]
    #
    #         weights = explained_variance_ratio / (np.sum(explained_variance_ratio) + 1e-12)
    #         aggregated = np.zeros((pca_normalized.shape[0],), dtype=np.float32)
    #         for i in range(pca_normalized.shape[1]):
    #             aggregated += pca_normalized[:, i] * weights[i]
    #
    #         num_points = coords_norm.shape[0]
    #         if num_points > max_points:
    #             sample_idx = np.random.choice(num_points, size=max_points, replace=False)
    #         else:
    #             sample_idx = np.arange(num_points)
    #
    #         coords_plot = coords_norm[sample_idx]
    #         rgb_plot = rgb[sample_idx]
    #
    #         safe_comment = (comment or "pca_feature").replace(" ", "_").replace("|", "_").replace("/", "_")
    #         batch_suffix = f"_b{batch_idx:02d}" if batch_size > 1 else ""
    #         image_path = os.path.join(output_dir, f"{safe_comment}{batch_suffix}.png")
    #         npz_path = os.path.join(output_dir, f"{safe_comment}{batch_suffix}.npz")
    #
    #         fig = plt.figure(figsize=(8, 8))
    #         ax = fig.add_subplot(111, projection="3d")
    #         ax.scatter(
    #             coords_plot[:, 0], coords_plot[:, 1], coords_plot[:, 2],
    #             c=rgb_plot,
    #             s=1.0,
    #             alpha=0.8,
    #             linewidths=0
    #         )
    #         title_comment = comment if comment is not None else "PCA Feature"
    #         ax.set_title(f"{title_comment}\nTop {n_components} cumulative={cumulative_variance:.1%}")
    #         ax.set_xlabel("x")
    #         ax.set_ylabel("y")
    #         ax.set_zlabel("z")
    #         ax.view_init(elev=28, azim=38)
    #         plt.tight_layout()
    #         plt.savefig(image_path, dpi=300, bbox_inches="tight")
    #         plt.close(fig)
    #
    #         np.savez_compressed(
    #             npz_path,
    #             coords=coords_norm,
    #             rgb=rgb,
    #             aggregated=aggregated,
    #             explained_variance_ratio=explained_variance_ratio
    #         )
    #
    #         log_comment = f"{comment} | batch={batch_idx}" if comment else f"batch={batch_idx}"
    #         _save_pca_log(log_path, explained_variance_ratio, cumulative_variance, log_comment)
    #         print(f"\t * Saved PCA point-cloud image to {image_path}.")
    #         print(f"\t * Saved PCA point-cloud data to {npz_path}.")
    #     print(f"\t * Saved PCA statistics to {log_path}.")

    def _forward_with_features(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        # out = self.estimator(x, t, cond)
        # if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        #     return out
        #
        # if not hasattr(self.estimator, "blocks"):
        #     raise TypeError(
        #         "Feature guidance expects a model that either returns `(pred, feature_dict)` "
        #         "or exposes `.blocks` with transformer blocks for hook-based feature capture."
        #     )
        if not torch.is_tensor(t):
            t = torch.tensor([1000 * t] * x.shape[0], device=x.device, dtype=torch.float32)
        elif t.ndim == 0:
            t = t.to(device=x.device, dtype=torch.float32).repeat(x.shape[0]) * 1000
        else:
            t = t.to(device=x.device, dtype=torch.float32)

        features: Dict[str, torch.Tensor] = {}
        hooks = []

        for block_idx, block in enumerate(self.estimator.blocks):
            block_prefix = f"block-{block_idx + 1:02d}"
            hooks.append(block.norm1.register_forward_hook(
                lambda _module, _inputs, output, key=f"{block_prefix}-norm-01": features.__setitem__(key, output)
            ))

            # hooks.append(block.norm1.register_forward_hook(
            #     lambda _module, _inputs, output, key=f"{block_prefix}-norm-01": (
            #         print(key, "hook grad_enabled =", torch.is_grad_enabled(),
            #               "requires_grad =", output.requires_grad,
            #               "grad_fn =", output.grad_fn),
            #         features.__setitem__(key, output)
            #     )[-1]
            # ))
            hooks.append(block.norm2.register_forward_hook(
                lambda _module, _inputs, output, key=f"{block_prefix}-norm-02": features.__setitem__(key, output)
            ))
            hooks.append(block.register_forward_hook(
                lambda _module, _inputs, output, key=f"{block_prefix}-final": features.__setitem__(key, output)
            ))

        try:
            pred = self.estimator(x, t, cond)
        finally:
            for hook in hooks:
                hook.remove()
        return pred, features

    # guidance
    # 用SparseStructureFlowModelWithFeatures算用于guidance的相似度
    def compute_guidance(
            self,
            mask_x0,
            mask_cur,
            mask_tar,
            mask_other,
            latent,
            latent_noise_ref,
            t,
            cond,
            up_ft_index=["block-17-norm-01", "block-17-norm-02"],
            energy_scale=2,
            w_edit=4,
            w_inpaint=0.2,
            w_content=6
    ):
        """

        Args:
            estimator: model
            mask_x0: R R R
            mask_cur: r r r
            mask_tar: r r r
            mask_other: b c r r r
            latent: b c r r r
            latent_noise_ref: b c r r r
            t:
            up_ft_index:
            up_scale: 不需要 estimator输出的只是和latent维度不同
            cond:
            energy_scale:
            w_edit:
            w_inpaint:
            w_content:
            dict_mask:

        Returns:

        """
        cos = nn.CosineSimilarity(dim=1)

        def _to_volume(feature_tokens: torch.Tensor, resolution: int) -> torch.Tensor:
            return feature_tokens.permute(0, 2, 1).reshape(feature_tokens.shape[0], feature_tokens.shape[2], resolution,
                                                           resolution, resolution)

        def _resize_mask(mask: torch.Tensor, spatial_size: Tuple[int, int, int],
                         ref_dtype: torch.dtype) -> torch.Tensor:
            if mask.dim() == 3:
                mask = mask[None, None]
            return (F.interpolate(mask.float(), size=spatial_size, mode="nearest") > 0).to(dtype=ref_dtype)

        with torch.no_grad():
            _, up_ft_tar_dict = self._forward_with_features(latent_noise_ref, t, cond)

        latent = latent.detach().requires_grad_(True)
        _, up_ft_cur_dict = self._forward_with_features(latent, t, cond)
        # print("up_ft_cur_dict:", up_ft_cur_dict.grad_fn)

        step_tag = float(t) if not torch.is_tensor(t) else float(t[0].item() / 1000.0)
        # self._save_hook_feature_grid(up_ft_tar_dict, step_tag=step_tag, source_tag="tar")
        # self._save_hook_feature_grid(up_ft_cur_dict, step_tag=step_tag, source_tag="cur")

        # 17 18层 或者
        feature_keys = sorted(k for k in up_ft_tar_dict.keys() if k in up_ft_index)
        if len(feature_keys) == 0:
            raise ValueError("SparseStructureFlowModelWithFeatures did not return any features.")

        # 中间特征的size上采样到统一大小
        resolution = round(up_ft_tar_dict[feature_keys[-1]].shape[1] ** (1 / 3))
        # target_size = (
        #     int(resolution * up_scale),
        #     int(resolution * up_scale),
        #     int(resolution * up_scale),
        # )

        up_ft_tar = []
        up_ft_cur = []
        for key in feature_keys:
            # print("up_ft_cur_dict[key]:", up_ft_cur_dict[key].requires_grad)
            # print("up_ft_tar_dict[key]:", up_ft_tar_dict[key].requires_grad)
            tar_ft = _to_volume(up_ft_tar_dict[key], resolution)
            cur_ft = _to_volume(up_ft_cur_dict[key], resolution)
            # print("cur_ft:", cur_ft.requires_grad)
            # print("tar_ft:", tar_ft.requires_grad)
            # if up_scale != 1:
            #     tar_ft = F.interpolate(tar_ft, size=target_size, mode="trilinear", align_corners=False)
            #     cur_ft = F.interpolate(cur_ft, size=target_size, mode="trilinear", align_corners=False)
            up_ft_tar.append(tar_ft)
            up_ft_cur.append(cur_ft)

            step_tag = float(t) if not torch.is_tensor(t) else float(t[0].item() / 1000.0)
            # self._visualize_pca_feature_pointcloud(
            #     cur_ft.detach(),
            #     comment=f"cur_{key}_t{step_tag:.4f}"
            # )
            # self._visualize_pca_feature_pointcloud(
            #     tar_ft.detach(),
            #     comment=f"tar_{key}_t{step_tag:.4f}"
            # )



        # 使用和latent相同的device和dtype
        loss_edit = 0
        for f_id in range(len(up_ft_tar)):
            for mask_cur_i, mask_tar_i in zip(mask_cur, mask_tar):
                # cur_mask = _resize_mask(mask_cur_i, up_ft_cur[f_id].shape[-3:], up_ft_cur[f_id].dtype).bool()
                # tar_mask = _resize_mask(mask_tar_i, up_ft_tar[f_id].shape[-3:], up_ft_tar[f_id].dtype).bool()

                cur_mask = mask_cur_i > 0.5
                tar_mask = mask_tar_i > 0.5

                up_ft_cur_vec = up_ft_cur[f_id][cur_mask.repeat(1, up_ft_cur[f_id].shape[1], 1, 1, 1)].view(
                    up_ft_cur[f_id].shape[1], -1).permute(1, 0)
                up_ft_tar_vec = up_ft_tar[f_id][tar_mask.repeat(1, up_ft_tar[f_id].shape[1], 1, 1, 1)].view(
                    up_ft_tar[f_id].shape[1], -1).permute(1, 0)
                num_pts = min(up_ft_cur_vec.shape[0], up_ft_tar_vec.shape[0])
                if num_pts > 0:
                    sim = (cos(up_ft_cur_vec[:num_pts], up_ft_tar_vec[:num_pts]) + 1.0) / 2.0
                    loss_edit = loss_edit + w_edit / (1 + 4 * sim.mean())
                    # print("sim grad：", sim.requires_grad)
                else:
                    raise ValueError(f"Invalid up_ft_cur_vec: {up_ft_cur_vec}")

                mask_overlap = ((cur_mask.float() + tar_mask.float()) > 1.5).float()
                mask_non_overlap = (tar_mask.float() - mask_overlap) > 0.5

                up_ft_cur_non_overlap = up_ft_cur[f_id][
                    mask_non_overlap.repeat(1, up_ft_cur[f_id].shape[1], 1, 1, 1)].view(up_ft_cur[f_id].shape[1],
                                                                                        -1).permute(1, 0)
                up_ft_tar_non_overlap = up_ft_tar[f_id][
                    mask_non_overlap.repeat(1, up_ft_tar[f_id].shape[1], 1, 1, 1)].view(up_ft_tar[f_id].shape[1],
                                                                                        -1).permute(1, 0)
                num_non_overlap = min(up_ft_cur_non_overlap.shape[0], up_ft_tar_non_overlap.shape[0])
                if num_non_overlap > 0:
                    sim_non_overlap = (cos(up_ft_cur_non_overlap[:num_non_overlap],
                                           up_ft_tar_non_overlap[:num_non_overlap]) + 1.0) / 2.0
                    loss_edit = loss_edit + w_inpaint * sim_non_overlap.mean()

        loss_con = 0
        other_mask = _resize_mask(mask_other, up_ft_tar[0].shape[-3:], up_ft_tar[0].dtype).bool()
        for f_id in range(len(up_ft_tar)):
            sim_other = (cos(up_ft_tar[f_id], up_ft_cur[f_id])[0][other_mask[0, 0]] + 1.0) / 2.0
            if sim_other.numel() > 0:
                loss_con = loss_con + w_content / (1 + 4 * sim_other.mean())

        loss_edit = loss_edit / len(up_ft_cur) / max(len(mask_cur), 1)
        loss_con = loss_con / len(up_ft_cur)

        # print("latent.requires_grad:", latent.requires_grad)
        #
        # print("latent.is_leaf:", latent.is_leaf)
        #
        # print("loss_edit.requires_grad:", loss_edit.requires_grad)
        #
        # print("loss_edit.grad_fn:", loss_edit.grad_fn)
        # print("loss_edit:", loss_edit)
        #
        # print("energy_scale type:", type(energy_scale))
        cond_grad_edit = torch.autograd.grad(loss_edit * energy_scale, latent, retain_graph=True)[0]
        cond_grad_con = torch.autograd.grad(loss_con * energy_scale, latent)[0]

        # 定义的编辑区域的mask 需要resize成cond_grad_edit的形状（latent的形状）
        mask = _resize_mask(mask_x0, cond_grad_edit.shape[-3:], latent.dtype)
        guidance = cond_grad_edit.detach() * 1e4 * mask + cond_grad_con.detach() * 1e4 * (1 - mask)
        self.estimator.zero_grad()
        # guidance维度 b c r r r
        # 预测的v是b c r r r
        return guidance

    def sde_once(
            self,
            model,
            x_t,
            t: float,
            t_prev: float,
            i: int,
            cond: Optional[Any] = None,
            **kwargs
    ):
        """
        Sample x_{t-1} using Euler / Euler-Maruyama.

        ODE mode:
            x_{t_prev} = x_t - (t - t_prev) * u

        SDE mode:
            dX_t = [
                (1 - sigma_t^2 * (1 - t) / (2t)) * u_t(X_t)
                - sigma_t^2 / (2t) * X_t
            ] dt + sigma_t dW_t

        where:
            u = pred_v + guidance
        """

        latent = x_t
        latent_noise_ref = kwargs.get("latent_noise_ref")[-(i + 1)]

        mask_x0 = kwargs.get("mask_x0")
        mask_cur = kwargs.get("mask_cur")
        mask_tar = kwargs.get("mask_tar")
        mask_other = kwargs.get("mask_other")

        # 可选参数
        SDE_strength = 0.4
        SDE_strength_un = 0.0
        alg = "D+"

        # 是否只在局部区域加 SDE
        use_regional_sde = True

        for key in [
            "latent_noise_ref",
            "mask_x0",
            "mask_cur",
            "mask_tar",
            "mask_other"
        ]:
            kwargs.pop(key, None)

        with torch.no_grad():
            pred_x_0, pred_eps, pred_v = self._get_model_prediction(
                model, x_t, t, cond, **kwargs
            )

        guidance = 0.0

        # 只在前 3/5 的采样过程中计算 guidance
        if i < 9:
            with torch.enable_grad():
                guidance = self.compute_guidance(
                    mask_x0,
                    mask_cur,
                    mask_tar,
                    mask_other,
                    latent,
                    latent_noise_ref,
                    t,
                    cond
                )

        def stat(name, x):
            if isinstance(x, float) or isinstance(x, int):
                print(f"{name}: scalar = {x}")
                return

            x_detach = x.detach()
            print(
                f"{name}: "
                f"shape={tuple(x_detach.shape)}, "
                f"mean={x_detach.mean().item():.6e}, "
                f"abs_mean={x_detach.abs().mean().item():.6e}, "
                f"norm={x_detach.norm().item():.6e}, "
                f"max_abs={x_detach.abs().max().item():.6e}"
            )

        stat("pred_v", pred_v)
        stat("guidance", guidance)

        if not isinstance(guidance, float):
            ratio = guidance.detach().norm() / (pred_v.detach().norm() + 1e-12)
            print(f"guidance / pred_v norm ratio: {ratio.item():.6e}")

        # ---------------------------------------------------------
        # 1. 构造 velocity / vector field
        # ---------------------------------------------------------
        u = pred_v + guidance

        # t, t_prev 可能是 float，也可能是 tensor，这里统一成 tensor
        if not torch.is_tensor(t):
            t_tensor = torch.tensor(t, device=x_t.device, dtype=x_t.dtype)
        else:
            t_tensor = t.to(device=x_t.device, dtype=x_t.dtype)

        if not torch.is_tensor(t_prev):
            t_prev_tensor = torch.tensor(t_prev, device=x_t.device, dtype=x_t.dtype)
        else:
            t_prev_tensor = t_prev.to(device=x_t.device, dtype=x_t.dtype)

        # 从 t 走到 t_prev，一般 t_prev < t
        dt = t_prev_tensor - t_tensor
        delta_t = t_tensor - t_prev_tensor

        # 防止 t 太接近 0 时除零
        eps = torch.tensor(1e-5, device=x_t.device, dtype=x_t.dtype)
        t_safe = torch.clamp(t_tensor, min=eps)

        # ---------------------------------------------------------
        # 2. 仿照原代码逻辑：只在 10 < i < 20 时启用 SDE
        # ---------------------------------------------------------
        if 3 < i < 6:
            eta_un = SDE_strength_un
            eta_rd = SDE_strength
        else:
            eta_un = 0.0
            eta_rd = 0.0

        # ---------------------------------------------------------
        # 3. 默认 ODE step
        #    sigma = 0 时，公式退化为：
        #    x_prev = x_t - (t - t_prev) * u
        # ---------------------------------------------------------
        pred_x_prev_ode = x_t - delta_t * u

        # ---------------------------------------------------------
        # 4. SDE step
        # ---------------------------------------------------------
        if (eta_rd > 0 or eta_un > 0) and alg == "D+":
            # sigma_t
            #
            # 这里先采用和你之前代码一致的设计：
            # - 非编辑区域使用 eta_un
            # - 编辑区域使用 eta_rd
            #
            # 如果你不想区分区域，可以直接令 sigma_t = eta_rd。
            sigma_un = torch.tensor(eta_un, device=x_t.device, dtype=x_t.dtype)
            sigma_rd = torch.tensor(eta_rd, device=x_t.device, dtype=x_t.dtype)

            def sde_step_with_sigma(sigma_t):
                sigma2 = sigma_t ** 2

                drift = (
                        (1.0 - sigma2 * (1.0 - t_safe) / (2.0 * t_safe)) * u
                        - sigma2 / (2.0 * t_safe) * x_t
                )

                noise = torch.randn_like(x_t)
                diffusion = sigma_t * torch.sqrt(torch.clamp(delta_t, min=0.0)) * noise

                # 因为是从 t 积分到 t_prev，所以 drift 乘 dt = t_prev - t
                return x_t + drift * dt + diffusion

            pred_x_prev_un = sde_step_with_sigma(sigma_un)
            pred_x_prev_rd = sde_step_with_sigma(sigma_rd)

            if use_regional_sde:
                # 优先使用 mask_other，因为你注释里说它和 latent 尺寸一致：
                # mask_other: b c 16 16 16
                if mask_other is not None:
                    mask = mask_other.to(device=x_t.device, dtype=x_t.dtype)

                    # 如果 mask 没有 channel 维，补到和 x_t 可广播
                    while mask.ndim < x_t.ndim:
                        mask = mask.unsqueeze(1)

                    if mask.shape[-len(x_t.shape[2:]):] != x_t.shape[2:]:
                        mask = F.interpolate(
                            mask.float(),
                            size=x_t.shape[2:],
                            mode="nearest"
                        ).to(dtype=x_t.dtype)

                    mask = (mask > 0).to(dtype=x_t.dtype)
                else:
                    # 如果没有 mask，就退化为全局 SDE
                    mask = torch.ones_like(x_t)

                # 区域 SDE：
                # - mask 区域使用 sigma_rd
                # - 非 mask 区域使用 sigma_un
                pred_x_prev = pred_x_prev_un * (1.0 - mask) + pred_x_prev_rd * mask

            else:
                # 全局 SDE
                pred_x_prev = pred_x_prev_rd

        else:
            # 没有 SDE 时，完全保持原来的 ODE 更新
            pred_x_prev = pred_x_prev_ode

        return edict({
            "pred_x_prev": pred_x_prev,
            "pred_x_0": pred_x_0
        })

    def edit_once(
            self,
            model,
            x_t,
            t: float,
            t_prev: float,
            i: int,
            cond: Optional[Any] = None,
            **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.

        Args:
            estimator:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            i: The index of the current sample.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """

        latent = x_t
        latent_noise_ref = kwargs.get("latent_noise_ref")[-(i + 1)]

        mask_x0 = kwargs.get("mask_x0")  # 编辑区域 d w h 16
        mask_cur = kwargs.get("mask_cur")  # target region
        mask_tar = kwargs.get("mask_tar")  # source region
        mask_other = kwargs.get("mask_other")  # bchw 更小的编辑区域 和latent一致 b c 16 16 16

        for key in ["latent_noise_ref", "mask_x0", "mask_cur", "mask_tar", "mask_other"]:
            kwargs.pop(key, None)

        with torch.no_grad():
            pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)

        guidance = 0.0

        # 只在前 3/5 的采样过程中计算 guidance
        if i < 9:
            with torch.enable_grad():
                guidance = self.compute_guidance(
                    mask_x0,
                    mask_cur,
                    mask_tar,
                    mask_other,
                    latent,
                    latent_noise_ref,
                    t,
                    cond
                )

        def stat(name, x):
            if isinstance(x, float) or isinstance(x, int):
                print(f"{name}: scalar = {x}")
                return

            x_detach = x.detach()
            print(
                f"{name}: "
                f"shape={tuple(x_detach.shape)}, "
                f"mean={x_detach.mean().item():.6e}, "
                f"abs_mean={x_detach.abs().mean().item():.6e}, "
                f"norm={x_detach.norm().item():.6e}, "
                f"max_abs={x_detach.abs().max().item():.6e}"
            )

        stat("pred_v", pred_v)
        stat("guidance", guidance)

        if not isinstance(guidance, float):
            ratio = guidance.detach().norm() / (pred_v.detach().norm() + 1e-12)
            print(f"guidance / pred_v norm ratio: {ratio.item():.6e}")

        pred_x_prev = x_t - (t - t_prev) * (pred_v + guidance)
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample_once(
            self,
            model,
            x_t,
            t: float,
            t_prev: float,
            cond: Optional[Any] = None,
            **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.

        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    # edit是每次sample once之后（此时已经cfg了）都要加上guidance 然后一次次迭代 （需要调用已有的sample once，所以写到这里）
    def edit(
            self,
            model,
            noise,
            cond: Optional[Any] = None,
            steps: int = 50,
            rescale_t: float = 1.0,
            verbose: bool = True,
            tqdm_desc: str = "Sampling",
            **kwargs
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            estimator:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            tqdm_desc: A customized tqdm desc.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        self._ensure_feature_estimator(model)
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        i = 0
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            out = self.edit_once(model, sample, t, t_prev, i, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
            i = i + 1
        ret.samples = sample
        return ret

    @torch.no_grad()
    def sample(
            self,
            model,
            noise,
            cond: Optional[Any] = None,
            steps: int = 50,
            rescale_t: float = 1.0,
            verbose: bool = True,
            tqdm_desc: str = "Sampling",
            **kwargs
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            tqdm_desc: A customized tqdm desc.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret

    @torch.no_grad()
    def inverse_sample(
            self,
            model,
            x_0,
            cond: Optional[Any] = None,
            steps: int = 50,
            rescale_t: float = 1.0,
            verbose: bool = True,
            tqdm_desc: str = "Inverting",
            **kwargs
    ):
        """
        Invert a latent from t=0 to t=1 with Euler method.

        Args:
            model: The flow model.
            x_0: The latent at t=0.
            cond: conditional information.
            steps: The number of inversion steps.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            tqdm_desc: A customized tqdm desc.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'samples': the inverted latent at t=1 (noise).
            - 'pred_x_t': a list of intermediate latent states.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = x_0
        t_seq = np.linspace(0, 1, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_next in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, sample, t, cond, **kwargs)
            sample = sample + (t_next - t) * pred_v
            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """

    @torch.no_grad()
    def sample(
            self,
            model,
            noise,
            cond,
            neg_cond,
            steps: int = 50,
            rescale_t: float = 1.0,
            guidance_strength: float = 3.0,
            verbose: bool = True,
            **kwargs
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond,
                              guidance_strength=guidance_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin,
                                       FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """

    @torch.no_grad()
    def sample(
            self,
            model,
            noise,
            cond,
            neg_cond,
            steps: int = 50,
            rescale_t: float = 1.0,
            guidance_strength: float = 3.0,
            guidance_interval: Tuple[float, float] = (0.0, 1.0),
            verbose: bool = True,
            **kwargs
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            guidance_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond,
                              guidance_strength=guidance_strength, guidance_interval=guidance_interval, **kwargs)

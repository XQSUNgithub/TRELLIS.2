from typing import *
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

    def _forward_with_features(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # out = self.estimator(x, t, cond)
        # if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        #     return out
        #
        # if not hasattr(self.estimator, "blocks"):
        #     raise TypeError(
        #         "Feature guidance expects a model that either returns `(pred, feature_dict)` "
        #         "or exposes `.blocks` with transformer blocks for hook-based feature capture."
        #     )

        features: Dict[str, torch.Tensor] = {}
        hooks = []

        for block_idx, block in enumerate(self.estimator.blocks):
            block_prefix = f"block-{block_idx + 1:02d}"
            hooks.append(block.norm1.register_forward_hook(
                lambda _module, _inputs, output, key=f"{block_prefix}-norm-01": features.__setitem__(key, output)
            ))
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
        energy_scale = 2,
        w_edit = 4,
        w_inpaint = 0.2,
        w_content = 6
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
            return feature_tokens.permute(0, 2, 1).reshape(feature_tokens.shape[0], feature_tokens.shape[2], resolution, resolution, resolution)

        def _resize_mask(mask: torch.Tensor, spatial_size: Tuple[int, int, int], ref_dtype: torch.dtype) -> torch.Tensor:
            if mask.dim() == 3:
                mask = mask[None, None]
            return (F.interpolate(mask.float(), size=spatial_size, mode="nearest") > 0).to(dtype=ref_dtype)

        with torch.no_grad():
            _, up_ft_tar_dict = self._forward_with_features(latent_noise_ref, t, cond)

        latent = latent.detach().requires_grad_(True)
        _, up_ft_cur_dict = self._forward_with_features(latent, t, cond)

        # 17 18层 或者
        feature_keys =sorted(k for k in up_ft_tar_dict.keys() if k in up_ft_index)
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
            tar_ft = _to_volume(up_ft_tar_dict[key], resolution)
            cur_ft = _to_volume(up_ft_cur_dict[key], resolution)
            # if up_scale != 1:
            #     tar_ft = F.interpolate(tar_ft, size=target_size, mode="trilinear", align_corners=False)
            #     cur_ft = F.interpolate(cur_ft, size=target_size, mode="trilinear", align_corners=False)
            up_ft_tar.append(tar_ft)
            up_ft_cur.append(cur_ft)

        # 使用和latent相同的device和dtype
        loss_edit = latent.new_tensor(0.0)
        for f_id in range(len(up_ft_tar)):
            for mask_cur_i, mask_tar_i in zip(mask_cur, mask_tar):
                # cur_mask = _resize_mask(mask_cur_i, up_ft_cur[f_id].shape[-3:], up_ft_cur[f_id].dtype).bool()
                # tar_mask = _resize_mask(mask_tar_i, up_ft_tar[f_id].shape[-3:], up_ft_tar[f_id].dtype).bool()

                up_ft_cur_vec = up_ft_cur[f_id][mask_cur_i.repeat(1, up_ft_cur[f_id].shape[1], 1, 1, 1)].view(up_ft_cur[f_id].shape[1], -1).permute(1, 0)
                up_ft_tar_vec = up_ft_tar[f_id][mask_tar_i.repeat(1, up_ft_tar[f_id].shape[1], 1, 1, 1)].view(up_ft_tar[f_id].shape[1], -1).permute(1, 0)
                num_pts = min(up_ft_cur_vec.shape[0], up_ft_tar_vec.shape[0])
                if num_pts > 0:
                    sim = (cos(up_ft_cur_vec[:num_pts], up_ft_tar_vec[:num_pts]) + 1.0) / 2.0
                    loss_edit = loss_edit + w_edit / (1 + 4 * sim.mean())

                mask_overlap = ((mask_cur_i.float() + mask_tar_i.float()) > 1.5).float()
                mask_non_overlap = (mask_tar_i.float() - mask_overlap) > 0.5

                up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1, up_ft_cur[f_id].shape[1], 1, 1, 1)].view(up_ft_cur[f_id].shape[1], -1).permute(1, 0)
                up_ft_tar_non_overlap = up_ft_tar[f_id][mask_non_overlap.repeat(1, up_ft_tar[f_id].shape[1], 1, 1, 1)].view(up_ft_tar[f_id].shape[1], -1).permute(1, 0)
                num_non_overlap = min(up_ft_cur_non_overlap.shape[0], up_ft_tar_non_overlap.shape[0])
                if num_non_overlap > 0:
                    sim_non_overlap = (cos(up_ft_cur_non_overlap[:num_non_overlap], up_ft_tar_non_overlap[:num_non_overlap]) + 1.0) / 2.0
                    loss_edit = loss_edit + w_inpaint * sim_non_overlap.mean()

        loss_con = latent.new_tensor(0.0)
        other_mask = _resize_mask(mask_other, up_ft_tar[0].shape[-3:], up_ft_tar[0].dtype).bool()
        for f_id in range(len(up_ft_tar)):
            sim_other = (cos(up_ft_tar[f_id], up_ft_cur[f_id])[0][other_mask[0, 0]] + 1.0) / 2.0
            if sim_other.numel() > 0:
                loss_con = loss_con + w_content / (1 + 4 * sim_other.mean())

        loss_edit = loss_edit / len(up_ft_cur) / max(len(mask_cur), 1)
        loss_con = loss_con / len(up_ft_cur)

        cond_grad_edit = torch.autograd.grad(loss_edit * energy_scale, latent, retain_graph=True)[0]
        cond_grad_con = torch.autograd.grad(loss_con * energy_scale, latent)[0]

        # 定义的编辑区域的mask 需要resize成cond_grad_edit的形状（latent的形状）
        mask = _resize_mask(mask_x0, cond_grad_edit.shape[-3:], latent.dtype)
        guidance = cond_grad_edit.detach() * 4e-2 * mask + cond_grad_con.detach() * 4e-2 * (1 - mask)
        self.estimator.zero_grad()
        #guidance维度 b c r r r
        # 预测的v是b c r r r
        return guidance

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
        with torch.no_grad():
         pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        latent = x_t
        latent_noise_ref = kwargs.get("latent_noise_ref")[-(i+1)]

        mask_x0 = kwargs.get("mask_x0") # 编辑区域 d w h 16
        mask_cur = kwargs.get("mask_cur") # target region
        mask_tar = kwargs.get("mask_tar") # source region
        mask_other = kwargs.get("mask_other") # bchw 更小的编辑区域 和latent一致 b c 16 16 16

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
        i=0
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            out = self.edit_once(model, sample, t, t_prev, i, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
            i=i+1
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
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
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
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, guidance_interval=guidance_interval, **kwargs)

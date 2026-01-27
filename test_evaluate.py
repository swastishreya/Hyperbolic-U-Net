from random import random
from PIL import Image
from skimage import morphology
import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import numpy as np
import gc
from tqdm import tqdm
from itertools import islice
from medpy.metric.binary import hd, hd95
import time

from losses import DiceLoss, BINARY_MODE, MULTICLASS_MODE
from metrics import get_stats, iou_score, sensitivity, specificity

from hypll.tensors import ManifoldTensor


def safe_hd(pred, true):
    # Ensure binary
    pred = (pred > 0).astype(np.bool_)
    true = (true > 0).astype(np.bool_)

    if np.sum(pred) == 0 and np.sum(true) == 0:
        return np.float64(0.0)
    elif np.sum(pred) == 0 or np.sum(true) == 0:
        return np.float64(1e6)
    return hd(pred, true)

def safe_hd95(pred, true):
    # Ensure binary
    pred = (pred > 0).astype(np.bool_)
    true = (true > 0).astype(np.bool_)

    if np.sum(pred) == 0 and np.sum(true) == 0:
        return np.float64(0.0)
    elif np.sum(pred) == 0 or np.sum(true) == 0:
        return np.float64(1e6)
    return hd95(pred, true)

def few_batches(dataloader, k_batches=10):
    return islice(dataloader, k_batches)

def fgsm_attack(model, images, true_masks, eps=0.01, loss_fn=None, device=None, targeted=False, target_masks=None):
    model.eval()
    device = device or images.device
    images_adv = images.clone().detach().to(device)
    images_adv.requires_grad = True 

    logits = model(images_adv)
    if targeted:
        assert target_masks is not None, "Provide target_masks for targeted attack"
        loss = loss_fn(logits, target_masks.long().to(device))
        # targeted: minimize loss -> gradient descent on image, so use sign(-grad)
        grad = torch.autograd.grad(loss, images_adv, retain_graph=False)[0]
        images_adv = images_adv - eps * grad.sign()
    else:
        loss = loss_fn(logits, true_masks.long().to(device))
        grad = torch.autograd.grad(loss, images_adv, retain_graph=False)[0]
        images_adv = images_adv + eps * grad.sign()

    images_adv = torch.clamp(images_adv.detach(), 0.0, 1.0)

    # ---- CLEANUP: delete large intermediates and free memory ----
    # remove grads on tensors we might have modified
    try:
        # remove references to big GPU tensors
        del logits, loss, grad
    except NameError:
        pass

    # ensure no gradient buffers on returned tensor
    if hasattr(images_adv, "grad") and images_adv.grad is not None:
        images_adv.grad = None

    # sync + collect + empty cache (only relevant for CUDA)
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    gc.collect()
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()

    return images_adv


def pgd_attack(model, images, true_masks, eps=0.03, alpha=0.007, iters=10, loss_fn=None, device=None, random_start=True, targeted=False, target_masks=None, cleanup_every=0):
    model.eval()
    device = device or images.device
    x_orig = images.clone().detach().to(device)
    if random_start:
        x = x_orig + (torch.empty_like(x_orig).uniform_(-eps, eps))
    else:
        x = x_orig.clone().detach()
    x.requires_grad_(True)

    for i in range(iters):
        logits = model(x)
        if targeted:
            assert target_masks is not None
            loss = loss_fn(logits, target_masks.to(device))
            # minimize => step opposite sign of grad
            grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
            x = x - alpha * torch.sign(grad)
        else:
            loss = loss_fn(logits, true_masks.to(device))
            grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
            x = x + alpha * torch.sign(grad)

        # projection step: clamp to l_inf ball around original
        x = torch.max(torch.min(x, x_orig + eps), x_orig - eps)
        x = torch.clamp(x, 0.0, 1.0)
        x = x.detach()
        x.requires_grad_(True)

        # ---- cleanup temporaries from this iteration ----
        # drop references to big tensors
        del logits, loss, grad

        # optionally do periodic cleanup to reduce fragmentation
        if cleanup_every and ((i + 1) % cleanup_every == 0):
            if device is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            gc.collect()
            if device is not None and device.type == "cuda":
                torch.cuda.empty_cache()

    return x.detach()


# -----------------------------
# Perturbation Functions
# -----------------------------
def add_motion_blur(x, kernel_size=9, angle=0.0):
    kernel = torch.zeros((kernel_size, kernel_size), device=x.device)
    kernel[kernel_size // 2, :] = 1.0
    kernel = kernel / kernel.sum()
    if angle != 0:
        kernel_img = kernel.unsqueeze(0).unsqueeze(0)
        kernel_img = TVF.rotate(kernel_img, angle)
        kernel = kernel_img.squeeze()
        kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(x.shape[1], 1, 1, 1)
    return torch.clamp(F.conv2d(x, kernel, groups=x.shape[1], padding=kernel_size // 2), 0., 1.)

def adjust_brightness(x, factor=0.2):
    return torch.clamp(x + factor, 0., 1.)

def adjust_contrast(x, factor=1.5):
    mean = x.mean(dim=(2, 3), keepdim=True)
    return torch.clamp((x - mean) * factor + mean, 0., 1.)

def add_gaussian_noise(x, sigma, generator=None):
    noise = (torch.randn(x.size(), generator=generator) * sigma).to(x.device)
    return torch.clamp(x + noise, 0., 1.)

def add_gaussian_noise_per_channel(x: torch.Tensor, sigma, generator=None):
    # sigma: float or sequence with len == x.shape[1] (channels)
    if isinstance(sigma, (list, tuple, torch.Tensor)):
        s = (torch.tensor(sigma, dtype=x.dtype, device=x.device).view(1, -1, *([1] * (x.dim()-2))))
        noise = (torch.randn(x.size(), device=x.device, generator=generator) * s)
    else:
        noise = (torch.randn(x.size(), device=x.device,generator=generator) * float(sigma)).to(x.device)
    return (x + noise).clamp(0., 1.)

def add_rician_noise(x: torch.Tensor, sigma: float, generator: torch.Generator = None) -> torch.Tensor:
    n_r = (torch.randn(x.size(), generator=generator) * sigma).to(x.device)
    n_i = (torch.randn(x.size(), generator=generator) * sigma).to(x.device)
    noisy = torch.sqrt((x + n_r)**2 + n_i**2)
    return noisy.clamp(0.0, 1.0)


def add_poisson_noise(x: torch.Tensor, peak: float = 1.0, generator: torch.Generator = None) -> torch.Tensor:
    # Scale to photon counts
    x = x.clamp(0.0, 1.0)
    scaled = x * torch.tensor(peak, dtype=x.dtype, device=x.device)
    if torch.any(scaled < 0) or torch.any(torch.isnan(scaled)) or torch.any(torch.isinf(scaled)):
        print("Invalid values in scaled:", scaled.min(), scaled.max())

    # Poisson sampling — must use float → integer counts
    noisy_counts = torch.poisson(scaled, generator=generator)
    # Rescale to [0,1]
    noisy = noisy_counts / peak
    return noisy.clamp(0.0, 1.0)

def add_speckle_noise(x, sigma=0.1, generator=None):
    noise = (torch.randn(x.size(), generator=generator) * sigma).to(x.device)
    return torch.clamp(x + x * noise, 0., 1.)

def add_speckle_noise_per_channel(x: torch.Tensor, sigma, generator=None):
    # sigma: float or sequence with len == x.shape[1] (channels)
    if isinstance(sigma, (list, tuple, torch.Tensor)):
        s = (torch.tensor(sigma, dtype=x.dtype, device=x.device).view(1, -1, *([1] * (x.dim()-2))))
        noise = (torch.randn(x.size(), device=x.device, generator=generator) * s)
    else:
        noise = (torch.randn(x.size(), device=x.device,generator=generator) * float(sigma)).to(x.device)
    return (x + x * noise).clamp(0., 1.)

def translate(image, mask, shift_x=32, shift_y=32):
    _, H, W = image.shape
    grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    yy, xx = grid[0], grid[1]
    xx = torch.clamp(xx - shift_x, 0, W-1)
    yy = torch.clamp(yy - shift_y, 0, H-1)
    shifted_img = image[:, yy, xx]
    shifted_mask = mask[yy, xx]
    return shifted_img, shifted_mask

def occlude(image, mask, occ_size=64, position="center"):
    _, H, W = image.shape
    img_occ = image.clone()
    
    if position == "center":
        top, left = (H - occ_size)//2, (W - occ_size)//2
    elif position == "tl":
        top, left = 0, 0
    elif position == "tr":
        top, left = 0, W - occ_size
    elif position == "bl":
        top, left = H - occ_size, 0
    elif position == "br":
        top, left = H - occ_size, W - occ_size
    
    img_occ[:, top:top+occ_size, left:left+occ_size] = 0.0
    return img_occ, mask


# -----------------------------
# Cropper + Aggregation
# -----------------------------
class TestTimeCropper:
    def __init__(self, resize_to=(256, 256), crop_size=150, crop_interval=12):
        self.resize_to = resize_to
        self.crop_size = crop_size
        self.crop_starts = np.arange(0, resize_to[0] - crop_size, crop_interval)

    def __call__(self, image, mask=None):
        image = TVF.resize(image, self.resize_to)
        if mask is not None:
            mask = TVF.resize(mask.unsqueeze(0), self.resize_to,
                              interpolation=TVF.InterpolationMode.NEAREST).squeeze(0)
        crops = []
        for start in self.crop_starts:
            cropped_img = TVF.crop(image, start, start, self.crop_size, self.crop_size)
            cropped_img = TVF.resize(cropped_img, self.resize_to)
            if mask is not None:
                cropped_mask = TVF.crop(mask, start, start, self.crop_size, self.crop_size)
                cropped_mask = TVF.resize(cropped_mask.unsqueeze(0), self.resize_to,
                                          interpolation=TVF.InterpolationMode.NEAREST).squeeze(0)
                crops.append((cropped_img, cropped_mask))
            else:
                crops.append((cropped_img, None))
        return crops

class ShiftCropper:
    def __init__(self, crop_size=224, positions=("center", "tl", "tr", "bl", "br")):
        self.crop_size = crop_size
        self.positions = positions

    def __call__(self, image, mask=None):
        _, H, W = image.shape
        crops = []

        for pos in self.positions:
            if pos == "center":
                top, left = (H - self.crop_size)//2, (W - self.crop_size)//2
            elif pos == "tl":
                top, left = 0, 0
            elif pos == "tr":
                top, left = 0, W - self.crop_size
            elif pos == "bl":
                top, left = H - self.crop_size, 0
            elif pos == "br":
                top, left = H - self.crop_size, W - self.crop_size
            else:
                continue

            img_crop = image[:, top:top+self.crop_size, left:left+self.crop_size]
            img_resized = F.interpolate(
                img_crop.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False
            ).squeeze(0)

            if mask is not None:
                mask_crop = mask[top:top+self.crop_size, left:left+self.crop_size]
                mask_resized = F.interpolate(
                    mask_crop.unsqueeze(0).unsqueeze(0).float(),
                    size=(H, W), mode="nearest"
                ).squeeze()
                crops.append((img_resized, mask_resized))
            else:
                crops.append((img_resized, None))

        return crops

def get_cropper_fn(cropper, params):
    resize_to, crop_size, crop_interval = params
    if "ttc" in cropper:
        return TestTimeCropper(resize_to=resize_to, crop_size=crop_size, crop_interval=crop_interval)
    elif "shift" in cropper:
        return ShiftCropper(crop_size=crop_size, positions=("center", "tl", "tr", "bl", "br"))
    return None

def foreground_aware_mean(crop_preds, threshold=0.1):
    crop_probs = torch.stack([p.softmax(dim=1) for p in crop_preds], dim=0)  # (N,C,H,W)
    fg_mask = (crop_probs.argmax(dim=1).sum(dim=(1,2)) > threshold).float()  # filter crops
    if fg_mask.sum() == 0:  # fallback: use normal mean
        return crop_probs.mean(dim=0, keepdim=True)
    weights = fg_mask.view(-1,1,1,1)
    return (crop_probs * weights).sum(dim=0, keepdim=True) / weights.sum()

def aggregate_crops(crop_preds, mode="mean", crop_weights=None):
    crop_probs = torch.stack([p.softmax(dim=1) for p in crop_preds], dim=0)
    if mode == "mean":
        return crop_probs.mean(dim=0, keepdim=True)
    elif mode == "max":
        return crop_probs.max(dim=0, keepdim=True).values
    elif mode == "majority":
        hard_preds = crop_probs.argmax(dim=1)
        votes = torch.stack([hard_preds == c for c in range(crop_probs.shape[1])], dim=1)
        votes = votes.sum(dim=0)
        maj = votes.argmax(dim=0, keepdim=True)
        onehot = torch.nn.functional.one_hot(maj[0], num_classes=crop_probs.shape[1])
        return onehot.permute(2, 0, 1).unsqueeze(0).float()
    elif mode == "weighted_mean":
        assert crop_weights is not None
        weights = torch.tensor(crop_weights, device=crop_probs.device).view(-1, 1, 1, 1)
        return (crop_probs * weights).sum(dim=0, keepdim=True) / weights.sum()
    elif mode == "foreground_mean":
        return foreground_aware_mean(crop_preds)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# -----------------------------
# Dispatcher for Perturbations
# -----------------------------
def apply_perturbation(images, perturbation, params, generator=None, model=None, loss_fn=None, device=None, true_masks=None):
    if perturbation is None:
        return images
    if "gaussian_noise" in perturbation:
        # return add_gaussian_noise(images, params, generator)
        return add_gaussian_noise_per_channel(images, params, generator)
    if "rician_noise" in perturbation:
        return add_rician_noise(images, params, generator)
    if "poisson_noise" in perturbation:
        return add_poisson_noise(images, params, generator)
    if "speckle_noise" in perturbation:
        return add_speckle_noise_per_channel(images, params, generator)
        # return add_speckle_noise(images, params, generator)
    if "brightness" in perturbation:
        return adjust_brightness(images, params)
    if "contrast" in perturbation:
        return adjust_contrast(images, params)
    if "motion_blur" in perturbation:
        k, angle = params
        return add_motion_blur(images, kernel_size=k, angle=angle)
    if "translate" in perturbation:
        shift_x, shift_y = params
        return torch.stack([translate(img, torch.zeros_like(img[0]), shift_x, shift_y)[0] for img in images])
    if "occlude" in perturbation:
        occ_size, pos = params
        return torch.stack([occlude(img, torch.zeros_like(img[0]), occ_size, pos)[0] for img in images])
    if "fgsm" in perturbation:
        assert model is not None and loss_fn is not None
        eps = params.get("eps", 0.01)
        return fgsm_attack(model, images, true_masks, eps=eps, loss_fn=loss_fn, device=device, targeted=params.get("targeted", False), target_masks=params.get("target_masks", None))
    if "pgd" in perturbation:
        assert model is not None and loss_fn is not None
        return pgd_attack(model, images, true_masks, params.get("eps",0.03), params.get("alpha",0.007), params.get("iters",10), loss_fn=loss_fn, device=device, random_start=params.get("random_start",True), targeted=params.get("targeted",False), target_masks=params.get("target_masks",None))
    raise ValueError(f"Unknown perturbation: {perturbation}")

def class_separation_metrics(
    features,
    mask,
    model,
    max_samples=1024,
    eps=1e-8,
):
    B, C, H, W = features.shape
    manifold = None

    # Flatten spatial dimensions
    if not isinstance(features, ManifoldTensor):
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
    else:
        try:
            manifold = model.manifolds[0]  # hyperbolic model
        except (AttributeError, IndexError):
            print("Hyperbolic model but must be single manifold")
            manifold = model.manifold
        features.tensor = features.tensor.permute(0, 2, 3, 1).reshape(B, H * W, C)
    mask = mask.reshape(B, H * W)

    inter_dists = []
    intra_dists = []
    ratios = []

    # --- subsample for stability ---
    def subsample(f):
        if f.size(0) > max_samples:
            idx = torch.randperm(f.size(0), device=f.device)[:max_samples]
            return f[idx]
        return f

    for b in range(B):
        f0 = features[b][mask[b] == 0]
        f1 = features[b][mask[b] == 1]

        if not isinstance(features, ManifoldTensor):
            if f0.numel() == 0 or f1.numel() == 0:
                zero = torch.tensor(0.0, device=features.device)
                inter_dists.append(zero)
                intra_dists.append(zero)
                ratios.append(zero)
                continue
            
            f0 = subsample(f0)
            f1 = subsample(f1)

            N0, N1 = f0.size(0), f1.size(0)

            # ---------- inter-class ----------
            f0_exp = f0[:, None, :].expand(N0, N1, C)
            f1_exp = f1[None, :, :].expand(N0, N1, C)

            inter = torch.cdist(f0_exp, f1_exp, p=2).mean()

            # ---------- intra-class ----------
            def intra_class_dist(f):
                N = f.size(0)
                if N < 2:
                    return torch.tensor(0.0, device=f.device)

                f_i = f[:, None, :].expand(N, N, C)
                f_j = f[None, :, :].expand(N, N, C)

                d = torch.cdist(f_i, f_j, p=2)
                # exclude diagonal
                return d[~torch.eye(N, dtype=torch.bool, device=d.device)].mean()

            intra0 = intra_class_dist(f0)
            intra1 = intra_class_dist(f1)
            intra = 0.5 * (intra0 + intra1)

            ratio = inter / (intra + eps)
        else:
            if f0.tensor.numel() == 0 or f1.tensor.numel() == 0:
                zero = torch.tensor(0.0, device=features.tensor.device)
                inter_dists.append(zero)
                intra_dists.append(zero)
                ratios.append(zero)
                continue

            f0 = subsample(f0.tensor)
            f1 = subsample(f1.tensor)

            N0, N1 = f0.size(0), f1.size(0)

            # ---------- inter-class ----------
            f0_exp = ManifoldTensor(
                data=f0[:, None, :].expand(N0, N1, C),
                manifold=manifold,
                man_dim=-1,
            )
            f1_exp = ManifoldTensor(
                data=f1[None, :, :].expand(N0, N1, C),
                manifold=manifold,
                man_dim=-1,
            )

            inter = manifold.dist(f0_exp, f1_exp).mean()

            # ---------- intra-class ----------
            def intra_class_dist(f):
                N = f.size(0)
                if N < 2:
                    return torch.tensor(0.0, device=f.tensor.device)

                f_i = ManifoldTensor(
                    data=f[:, None, :].expand(N, N, C),
                    manifold=manifold,
                    man_dim=-1,
                )
                f_j = ManifoldTensor(
                    data=f[None, :, :].expand(N, N, C),
                    manifold=manifold,
                    man_dim=-1,
                )

                d = manifold.dist(f_i, f_j)
                # exclude diagonal
                return d[~torch.eye(N, dtype=torch.bool, device=d.device)].mean()

            intra0 = intra_class_dist(f0)
            intra1 = intra_class_dist(f1)
            intra = 0.5 * (intra0 + intra1)

            ratio = inter / (intra + eps)

        inter_dists.append(inter)
        intra_dists.append(intra)
        ratios.append(ratio)

    inter_dists = torch.stack(inter_dists)
    intra_dists = torch.stack(intra_dists)
    ratios = torch.stack(ratios)

    batch_metrics = {
    "inter_class_mean": inter_dists.mean(),
    "inter_class_std": inter_dists.std(unbiased=False),
    "intra_class_mean": intra_dists.mean(),
    "intra_class_std": intra_dists.std(unbiased=False),
    "mean_separation_ratio": ratios.mean(),
    "separation_ratio_std": ratios.std(unbiased=False),
    "ratio_of_means": inter_dists.mean() / (intra_dists.mean() + eps)
    }

    per_image_metrics = {
        "inter_class": inter_dists,
        "intra_class": intra_dists,
        "separation_ratio": ratios,
    }

    return batch_metrics, per_image_metrics


# -----------------------------
# Unified Evaluation
# -----------------------------
@torch.inference_mode()
def test_evaluate(model, dataloader, device, amp,
                  perturbation=None, perturb_params=None, generator=None,
                  cropper=None, cropper_params=None, crop_agg="mean",
                  loss_fn_adv=None, max_samples=2, save_dir="evaluation_visuals", model_type=None):
    model.eval()
    num_val_batches = len(dataloader)

    sample_count = 0
    os.makedirs(save_dir, exist_ok=True)
    if model.n_classes == 1:
        loss_mode = BINARY_MODE
    else:
        loss_mode = MULTICLASS_MODE
    loss_fn = DiceLoss(loss_mode, from_logits=True)

    if loss_fn_adv is None:
        loss_fn_adv = torch.nn.CrossEntropyLoss()

    dice_loss, hd_dist, hd95_dist = 0, 0, 0
    tot_tp, tot_fp, tot_fn, tot_tn = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    times = []

    if cropper is not None:
        cropper = get_cropper_fn(cropper, cropper_params)

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Evaluation', unit='batch', leave=False):
            start = time.time()
            images, true_masks, image_names = batch['image'], batch['mask'], batch['image_name']
            images = images.to(device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device, dtype=torch.long)

            # Apply perturbation
            images = apply_perturbation(images, perturbation, perturb_params, generator, model, loss_fn, device, true_masks)

            # Cropping flow
            if cropper is not None:
                batch_preds, batch_targets = [], []
                for i in range(images.size(0)):
                    crops = cropper(images[i], true_masks[i])
                    crop_preds = [model(crop_img.unsqueeze(0).to(device)) for crop_img, _ in crops]
                    agg_pred = aggregate_crops(crop_preds, mode=crop_agg)
                    batch_preds.append(agg_pred.squeeze(0))
                    batch_targets.append(true_masks[i].unsqueeze(0))
                logits_masks = torch.cat(batch_preds, dim=0)
                true_masks = torch.cat(batch_targets, dim=0)
            else:
                logits_masks = model(images)

            end = time.time()
            times.append(end - start)
   

            # Metrics
            dice_loss += loss_fn(logits_masks, true_masks)
            prob_masks = logits_masks.sigmoid() if model.n_classes == 1 else logits_masks.softmax(dim=1)
            pred_masks = (prob_masks > 0.5).long() if model.n_classes == 1 else prob_masks.argmax(dim=1)

            if sample_count < max_samples:
                batch_to_save = min(max_samples - sample_count, images.size(0))
                save_batch_visuals(images[:batch_to_save], true_masks[:batch_to_save],
                                pred_masks[:batch_to_save], image_names[:batch_to_save],
                                save_dir, perturbation=perturbation)
                sample_count += batch_to_save
            else:
                continue

            tp, fp, fn, tn = get_stats(pred_masks, true_masks,
                                       mode="binary" if model.n_classes == 1 else "multiclass",
                                       num_classes=model.n_classes if model.n_classes > 1 else None)
            tot_tp, tot_fp, tot_fn, tot_tn = torch.cat([tp, tot_tp]), torch.cat([fp, tot_fp]), torch.cat([fn, tot_fn]), torch.cat([tn, tot_tn])

            pred_np, true_np = pred_masks.cpu().numpy(), true_masks.cpu().numpy()
            hausdorff1, hausdorff2 = safe_hd(pred_np, true_np), safe_hd(true_np, pred_np)
            hausdorff95_1, hausdorff95_2 = safe_hd95(pred_np, true_np), safe_hd95(true_np, pred_np)
            hd_dist += max(hausdorff1, hausdorff2)
            hd95_dist += max(hausdorff95_1, hausdorff95_2)
        
        times = np.array(times)

    results = {
        "dice_loss": dice_loss.mean().cpu().item() / max(num_val_batches, 1),
        "hausdorff": hd_dist / max(num_val_batches, 1),
        "hausdorff_95": hd95_dist / max(num_val_batches, 1),
        "per_image_iou": iou_score(tot_tp, tot_fp, tot_fn, tot_tn, reduction="micro-imagewise").cpu().item(),
        "dataset_iou": iou_score(tot_tp, tot_fp, tot_fn, tot_tn, reduction="micro").cpu().item(),
        "dataset_sensitivity": sensitivity(tot_tp, tot_fp, tot_fn, tot_tn, reduction="macro-imagewise").cpu().item(),
        "dataset_specificity": specificity(tot_tp, tot_fp, tot_fn, tot_tn, reduction="macro-imagewise").cpu().item(),
    }
    return results

def save_batch_visuals(images, true_masks, pred_masks, names, save_dir, perturbation=None):
    """
    Save a batch of images, ground truth, predictions, error maps, and overlays.
    Uses PIL for saving images.
    """
    images = images.detach().cpu()
    true_masks = true_masks.detach().cpu()
    pred_masks = pred_masks.detach().cpu()

    for i in range(images.size(0)):
        img = images[i]
        gt = true_masks[i]
        pred = pred_masks[i]
        name = names[i] if isinstance(names[i], str) else f"sample_{i:03d}"

        # Prepare arrays
        if img.size(0) == 3:
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            img_np = (img.squeeze(0).numpy() * 255).astype(np.uint8)
            img_np = np.stack([img_np] * 3, axis=-1)

        gt_np = gt.numpy().astype(np.uint8)
        pred_np = pred.numpy().astype(np.uint8)

        # Lesion area percentages
        lesion_area_gt = 100 * gt_np.sum() / gt_np.size
        lesion_area_pred = 100 * pred_np.sum() / pred_np.size

        # Difference map
        diff_map = np.abs(gt_np - pred_np)

        # Boundaries
        gt_boundary = morphology.binary_dilation(gt_np) ^ gt_np
        pred_boundary = morphology.binary_dilation(pred_np) ^ pred_np

        # Overlay boundaries on image
        overlay = img_np.copy()
        overlay[gt_boundary > 0] = [0, 255, 0]    # GT = green
        overlay[pred_boundary > 0] = [255, 0, 0]  # Pred = red

        # Save images
        base_name = os.path.splitext(name)[0]
        Image.fromarray(img_np).save(os.path.join(save_dir, f"{perturbation}_{base_name}_image.png"))
        Image.fromarray((gt_np * 255).astype(np.uint8)).save(os.path.join(save_dir, f"{perturbation}_{base_name}_gt.png"))
        Image.fromarray((pred_np * 255).astype(np.uint8)).save(os.path.join(save_dir, f"{perturbation}_{base_name}_pred.png"))
        Image.fromarray((diff_map * 255).astype(np.uint8)).save(os.path.join(save_dir, f"{perturbation}_{base_name}_error.png"))
        Image.fromarray(overlay.astype(np.uint8)).save(os.path.join(save_dir, f"{perturbation}_{base_name}_overlay.png"))

        print(f"[{base_name}] GT lesion area: {lesion_area_gt:.2f}% | Pred lesion area: {lesion_area_pred:.2f}%")
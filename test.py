import argparse
import logging
import torch
import json
from pprint import pformat
import numpy as np
import random

from test_evaluate import test_evaluate
from dataloading.make_dataset import MakeDataset
from unet.euclidean_unet.model import UNet
from unet.euclidean_unet.nested_unet import NUNet, NestedUNet
from unet.hyperbolic_unet.model import FlexHUNet, HUNet
from unet.hyperbolic_unet.nested_unet import HNUNet, HNestedUNet


# ------------------ Utils ------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_grad_status(module, inp, out):
    print(f"{module.__class__.__name__}: out.requires_grad={out.requires_grad}")

def build_model(args, model_type, device):
    """Factory to build and return the correct model."""
    if model_type == 'euc':
        model = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear, init_feats=args.init_feats, depth=args.depth)
    elif model_type == 'hyp':
        if len(args.curvature) == 1:
            model = HUNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear, curvature=args.curvature[0], trainable=args.trainable, init_feats=args.init_feats, depth=args.depth)
        else:
            assert len(args.curvature) == (args.depth+1)
            model = FlexHUNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear, curvature=args.curvature, trainable=args.trainable, init_feats=args.init_feats, depth=args.depth)
    elif model_type == 'nunet':
        model = NUNet(n_channels=args.channels, n_classes=args.classes, init_feats=args.init_feats)
    elif model_type == 'hnunet':
        model = HNUNet(n_channels=args.channels, n_classes=args.classes, curvature=args.curvature[0], trainable=args.trainable, init_feats=args.init_feats)
    elif model_type == 'nestedunet':
        model = NestedUNet(n_channels=args.channels, n_classes=args.classes, init_feats=args.init_feats, deep_supervision=False)
    elif model_type == 'hnestedunet':
        model = HNestedUNet(n_channels=args.channels, n_classes=args.classes, curvature=args.curvature[0], trainable=args.trainable, init_feats=args.init_feats, deep_supervision=False)
    else:
        raise NotImplementedError('Specify a model among euc, hyp, lhyp or mmu')
    model = model.to(memory_format=torch.channels_last)

    model = model.to(device=device, memory_format=torch.channels_last)
    return model


# ------------------ Evaluation ------------------

def evaluate_model(args, dataset_path, model_type, weight_path, device, batch_size, amp=False, msk_scale=1.0, croppers=None, cropper_params=None, perturbations=None, perturb_params=None, max_samples=10, save_dir=None):
    """Build, load weights, evaluate across perturbations, return metrics."""

    # Build dataloader (same order for all perturbations)
    dataset = MakeDataset(dataset_path, args.scale, batch_size, val_percent=0.1, msk_scale=msk_scale)
    _, _, test_loader = dataset.get_loaders()

    # Build model
    model = build_model(args, model_type, device)
    state_dict = torch.load(weight_path, map_location=device)
    if "mask_values" in state_dict:
        del state_dict["mask_values"]
    model.load_state_dict(state_dict)
    logging.info(f"Loaded {model_type} from {weight_path}")
    metrics = test_evaluate(
        model, test_loader, device, amp,
        cropper=None, perturbation=None,
        max_samples=max_samples, save_dir=save_dir,
        model_type=model_type
    )

    metrics['dice_score'] = 1 - metrics['dice_loss']
    del model
    torch.cuda.empty_cache()

    return metrics


# ------------------ Main ------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True)
    parser.add_argument('--channels', '-inc', type=int, default=1)
    parser.add_argument('--classes', '-c', type=int, default=2)
    parser.add_argument('--experiment_name', '-exp', required=True, help="Experiment name")
    parser.add_argument('--model', required=True, help="Model type (euc/hyp/...)")
    parser.add_argument('--load', required=True, help="Path to weights for model")
    parser.add_argument('--init_feats', '-inft', type=int, default=8, help='Initial number of feature maps of the network')
    parser.add_argument('--depth', '-dpth', type=int, default=4, help='Depth of the network')
    parser.add_argument('--batch-size', '-b', type=int, default=1)
    parser.add_argument('--scale', '-s', type=float, default=0.5)
    parser.add_argument('--mask-scale', '-ms', type=float, default=0.5)
    parser.add_argument('--curvature', '-curv', nargs='+', type=float , help='Set curvatures of the manifolds of the hyperbolic model')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--bilinear', action='store_true')
    parser.add_argument('--trainable', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    set_seed(42)
    args = get_args()
    print("Arguments:", pformat(vars(args)))

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    results = {}

    # Second model
    results["hyp"] = evaluate_model(
        args, args.dataset, args.model, args.load,
        device=device, batch_size=args.batch_size,
        amp=args.amp, msk_scale=args.mask_scale,
        croppers=None, cropper_params=None,
        perturbations=None, perturb_params=None,
        max_samples=4, save_dir=f"hyp_{args.experiment_name}"
    )

    # Save
    with open(f"{args.experiment_name}.json", "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {args.experiment_name}.json")

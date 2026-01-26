import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path
from torch import optim
from tqdm import tqdm
import time

import wandb

from hypll.optim import RiemannianAdam, RiemannianSGD

from evaluate import evaluate
from dataloading.make_dataset import MakeDataset
from unet.euclidean_unet.model import UNet
from unet.euclidean_unet.nested_unet import NUNet, NestedUNet
from unet.hyperbolic_unet.model import FlexHUNet, HUNet
from unet.hyperbolic_unet.nested_unet import HNUNet, HNestedUNet
# from unet.hyperbolic_unet.unet_parts import register_norm_hooks as hyp_register_norm_hooks
# from unet.euclidean_unet.unet_parts import register_norm_hooks as euc_register_norm_hooks
from weight_init_autograd_util import init_then_hyperbolic_scale
from losses import *
from metrics import *


def train_model(
        args,
        dataset_path,
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        msk_scale: float = 0.5,
):
    # Create dataset
    dataset_name = (dataset_path.split('/'))[-1]
    dir_checkpoint = Path('./checkpoints/' + dataset_name)
    dataset = MakeDataset(dataset_path, img_scale, batch_size, val_percent, msk_scale)
    train_loader, val_loader, test_loader = dataset.get_loaders()

    if 'euc' in args.model:
        name = 'd={}-unet={}-optim={}-lr={}-bs={}-loss={}-alpha={}-beta={}-init_feats={}-depth={}'.format(dataset_name, args.model, args.optim, args.lr, args.batch_size, args.loss, args.alpha, args.beta, args.init_feats, args.depth)
    elif 'hyp' in args.model:
        name = 'd={}-unet={}-optim={}-lr={}-bs={}-curv={}-trainable={}-loss={}-alpha={}-beta={}-init_feats={}-depth={}'.format(dataset_name, args.model, args.optim, args.lr, args.batch_size, args.curvature, args.trainable, args.loss, args.alpha, args.beta, args.init_feats, args.depth)
    elif 'nunet' in args.model:
        name = 'd={}-unet={}-optim={}-lr={}-bs={}-loss={}-alpha={}-beta={}-init_feats={}-depth={}'.format(dataset_name, args.model, args.optim, args.lr, args.batch_size, args.loss, args.alpha, args.beta, args.init_feats, args.depth)
    elif 'hnunet' in args.model:
        name = 'd={}-unet={}-optim={}-lr={}-bs={}-curv={}-trainable={}-loss={}-alpha={}-beta={}-init_feats={}-depth={}'.format(dataset_name, args.model, args.optim, args.lr, args.batch_size, args.curvature[0], args.trainable, args.loss, args.alpha, args.beta, args.init_feats, args.depth)
    elif 'nestedunet' in args.model:
        name = 'd={}-unet={}-optim={}-lr={}-bs={}-loss={}-alpha={}-beta={}-init_feats={}-depth={}'.format(dataset_name, args.model, args.optim, args.lr, args.batch_size, args.loss, args.alpha, args.beta, args.init_feats, args.depth)
    elif 'hnestedunet' in args.model:
        name = 'd={}-unet={}-optim={}-lr={}-bs={}-curv={}-trainable={}-loss={}-alpha={}-beta={}-init_feats={}-depth={}'.format(dataset_name, args.model, args.optim, args.lr, args.batch_size, args.curvature[0], args.trainable, args.loss, args.alpha, args.beta, args.init_feats, args.depth)
    id_name = str(int(time.time()))

    # Initialize logging
    experiment = wandb.init(project=args.project, resume='allow', name=name, id=id_name)
    experiment.config.update(
        dict(model=model.__class__.__name__, 
             epochs=epochs, 
             batch_size=batch_size, 
             learning_rate=learning_rate,
             val_percent=val_percent, 
             save_checkpoint=save_checkpoint, 
             img_scale=img_scale, 
             amp=amp,
             dataset=dataset_name,
             optim=args.optim,
             curvature=args.curvature,
             trainable=args.trainable,
             bilinear=args.bilinear,
             channels=args.channels,
             classes=args.classes,
             loss=args.loss,
             alpha=args.alpha,
             beta=args.beta,
             init_feats=args.init_feats,
             init_mode=args.init_mode,
             depth=args.depth),
        allow_val_change=True
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_loader.dataset)}
        Validation size: {len(val_loader.dataset)}
        Test size:       {len(test_loader.dataset)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if args.optim == 'adam':
        optimizer = RiemannianAdam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif args.optim == 'sgd':
        optimizer = RiemannianSGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError('Specify an optimizer among Adam or SGD')

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Setup loss function
    loss_mode = None
    if args.classes == 1:
        loss_mode = BINARY_MODE
        ce_like_loss_fn = nn.BCEWithLogitsLoss()
    elif args.classes > 1:
        loss_mode = MULTICLASS_MODE
        ce_like_loss_fn = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError('Specify a loss mode among BINARY_MODE or MULTICLASS_MODE')

    # Dice and Tversky losses are alike
    # CE and Focal losses are alike
    """
    Default setup used in the original Focal Loss paper on RetinaNet (object detection):
    alpha = 0.25  # positive class weight
    gamma = 2.0   # focus strength

    For a Focal Tversky Loss the following setup can be used
    alpha = 0.3   # false positives weight
    beta = 0.7    # false negatives weight
    gamma = 1.33  # focus on hard examples

    For tiny objects use:
    alpha = 0.2
    beta = 0.8
    gamma = 1.5
    """
    if 'dice' in args.loss:
        loss_fn = DiceLoss(loss_mode, from_logits=True)
    elif 'tversky' in args.loss:
        loss_fn = TverskyLoss(loss_mode, from_logits=True, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    elif 'focal' in args.loss:
        ce_like_loss_fn = FocalLoss(alpha=args.alpha, gamma=args.gamma)

    global_step = 0
    val_score_prev = 0

    if args.init_mode is not None and args.init_mode != 'none' and args.model.startswith('h'):
        scales = init_then_hyperbolic_scale(
                model,
                train_loader,   # or just pass 'loader' and it will take batch[0]
                init_mode=args.init_mode,  # "kaiming" or "orthogonal"
                nonlinearity="relu",
                batches=args.batch_size,  # number of batches to use for the initialization
                newton_iters=10,
                device=device
            )
        print(scales)

    # Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks, masks_mean = batch['image'], batch['mask'], batch['mask_mean']

                if args.model != 'lhyp':
                    assert images.shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'
                else:
                    assert images.shape[1] == (model.n_channels-1), \
                        f'Lorentz network has been defined with {model.n_channels} input channels, which should be one' \
                        f'more channel than the images but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                masks_mean = masks_mean + torch.ones_like(masks_mean)
                masks_mean = masks_mean.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    logits_masks = model(images)
                    if model.n_classes == 1:
                        logits_masks = logits_masks.contiguous()

                        loss = loss_fn(logits_masks, true_masks)
                        loss += ce_like_loss_fn(logits_masks, true_masks)

                        prob_masks = logits_masks.sigmoid()
                        pred_masks = (prob_masks > 0.5).float()
                    else:
                        assert logits_masks.shape[1] == model.n_classes
                        logits_masks = logits_masks.contiguous()

                        loss = loss_fn(logits_masks, true_masks)
                        loss += ce_like_loss_fn(logits_masks, true_masks)

                        prob_masks = logits_masks.softmax(dim=1)
                        pred_masks = prob_masks.argmax(dim=1)                        

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                curv_val = {}
                if args.curvature is not None:
                    if args.model == 'hyp':
                        if len(args.curvature) == 1:
                            curv_val['curv1'] = model.manifold.c()
                        elif len(args.curvature) > 1:
                            idx = 1
                            for man in model.manifolds:
                                curv_val['curv'+str(idx)] = man.c()
                                idx += 1
                    elif args.model == 'lhyp':
                        curv_val['curv1'] = model.manifold.k.item()
                    elif args.model == 'mmu' or args.model == 'hnunet' or args.model == 'hnestedunet':
                        curv_val['curv1'] = model.manifold.c()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch,
                    **curv_val

                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (len(train_loader.dataset) // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value.data) | torch.isnan(value.data)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if (value.grad is not None) and (not (torch.isinf(value.grad) | torch.isnan(value.grad)).any()):
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_metrics = evaluate(model, val_loader, device, amp)
                        val_dice_score = 1 - val_metrics['dice_loss']
                        scheduler.step(val_dice_score)

                        logging.info('Validation Dice score: {}'.format(val_dice_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation dice': val_dice_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(pred_masks[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **val_metrics,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint and val_dice_score > val_score_prev:
            val_score_prev = val_dice_score
            Path(dir_checkpoint / args.model).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / args.model / '{}-epoch{}.pth'.format(id_name, epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

        if epoch % 5 == 0:
            test_metrics = evaluate(model, test_loader, device, amp)
            test_dice_score = 1 - test_metrics['dice_loss']
            logging.info('Test Dice score: {}'.format(test_dice_score))
            try:
                experiment.log({
                    'test dice': test_dice_score,
                    'step': global_step,
                    'epoch': epoch,
                    **test_metrics
                })
            except:
                pass


def get_args():
    parser = argparse.ArgumentParser(description='Train the different UNets on images and target masks')
    parser.add_argument('--project', '-p',  type=str, help='Project name')
    parser.add_argument('--dataset', '-d',  type=str, help='Path to the dataset')
    parser.add_argument('--channels', '-inc',  type=int, default=1, help='Number of input channels')
    parser.add_argument('--model', '-m', choices=['euc', 'hyp', 'lhyp', 'mmu', 'nunet', 'hnunet', 'nestedunet', 'hnestedunet'], help='Type of model (hyp/euc)')
    parser.add_argument('--init_mode', '-inmo', choices=['kaiming', 'ortho', 'none'], help='Type of model (kaiming/ortho)')
    parser.add_argument('--init_feats', '-inft', type=int, default=32, help='Initial number of feature maps of the network')
    parser.add_argument('--depth', '-dpth', type=int, default=4, help='Depth of the network')
    parser.add_argument('--optim', '-o', choices=['adam', 'sgd'], help='Type of optimizer (Adam/SGD)')
    parser.add_argument('--loss', '-los', choices=['dice+CE', 'dice+focal', 'tversky+CE', 'tversky+focal'], help='Type of losses (Combinations of dice, cross entropy, focal, etc.)')
    parser.add_argument('--alpha', '-alp', type=float, default=0.3, help='Alpha value associated with the loss') # 0.2 
    parser.add_argument('--beta', '-bta', type=float, default=0.7, help='Beta value associated with the loss') # 0.8
    parser.add_argument('--gamma', '-gma', type=float, default=1.33, help='Gamma value associated with the loss') # 1.5
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--mask-scale', '-ms', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--curvature', '-curv', nargs='+', type=float , help='Set curvatures of the manifolds of the hyperbolic model')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--trainable', action='store_true', default=False, help='Use to make manifold of the hyperbolic model trainable')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if args.model == 'euc':
        model = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear, init_feats=args.init_feats, depth=args.depth)
    elif args.model == 'hyp':
        if len(args.curvature) == 1:
            model = HUNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear, curvature=args.curvature[0], trainable=args.trainable, init_feats=args.init_feats, depth=args.depth)
        else:
            assert len(args.curvature) == (args.depth+1)
            model = FlexHUNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear, curvature=args.curvature, trainable=args.trainable, init_feats=args.init_feats, depth=args.depth)
    elif args.model == 'nunet':
        model = NUNet(n_channels=args.channels, n_classes=args.classes, init_feats=args.init_feats)
    elif args.model == 'hnunet':
        model = HNUNet(n_channels=args.channels, n_classes=args.classes, curvature=args.curvature[0], trainable=args.trainable, init_feats=args.init_feats)
    elif args.model == 'nestedunet':
        model = NestedUNet(n_channels=args.channels, n_classes=args.classes, init_feats=args.init_feats, deep_supervision=False)
    elif args.model == 'hnestedunet':
        model = HNestedUNet(n_channels=args.channels, n_classes=args.classes, curvature=args.curvature[0], trainable=args.trainable, init_feats=args.init_feats, deep_supervision=False)
    else:
        raise NotImplementedError('Specify a model among euc, hyp, lhyp or mmu')
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            args=args,
            dataset_path=args.dataset,
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            save_checkpoint=True,
            msk_scale=args.mask_scale,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            args=args,
            dataset_path=args.dataset,
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            save_checkpoint=True,
            msk_scale=args.mask_scale
        )

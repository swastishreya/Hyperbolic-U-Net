import torch
from tqdm import tqdm

from losses import *
from metrics import *


@torch.inference_mode()
def evaluate(model, dataloader, device, amp):
    model.eval()
    num_val_batches = len(dataloader)
    dice_loss = 0
    tot_tp, tot_fp, tot_fn, tot_tn = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    if model.n_classes == 1:
        loss_mode = BINARY_MODE
    elif model.n_classes > 1:
        loss_mode = MULTICLASS_MODE
    else:
        raise NotImplementedError('Specify a loss mode among BINARY_MODE or MULTICLASS_MODE')
    loss_fn = DiceLoss(loss_mode, from_logits=True)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            images, true_masks = batch['image'], batch['mask']

            # move images and labels to correct device and type
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            # predict the mask
            logits_masks = model(images)

            if model.n_classes == 1:
                dice_loss += loss_fn(logits_masks, true_masks)
                logits_masks = logits_masks.contiguous()

                prob_masks = logits_masks.sigmoid()
                pred_masks = (prob_masks > 0.5).float()
                tp, fp, fn, tn = get_stats(
                    pred_masks.long(), true_masks.long(), mode="binary"
                )
            else:
                assert logits_masks.shape[1] == model.n_classes
                logits_masks = logits_masks.contiguous()

                dice_loss += loss_fn(logits_masks, true_masks)

                prob_masks = logits_masks.softmax(dim=1)
                pred_masks = prob_masks.argmax(dim=1)

                tp, fp, fn, tn = get_stats(
                    pred_masks, true_masks, mode="multiclass", num_classes=model.n_classes
                )

            tot_tp = torch.cat([tp, tot_tp])
            tot_fp = torch.cat([fp, tot_fp])
            tot_fn = torch.cat([fn, tot_fn])
            tot_tn = torch.cat([tn, tot_tn])
            
        # Per-image IoU and dataset IoU calculations
        per_image_iou = iou_score(tot_tp, tot_fp, tot_fn, tot_tn, reduction="micro-imagewise")
        dataset_iou = iou_score(tot_tp, tot_fp, tot_fn, tot_tn, reduction="micro")
        dataset_sensitivity = sensitivity(tot_tp, tot_fp, tot_fn, tot_tn, reduction="macro")
        dataset_specificity = specificity(tot_tp, tot_fp, tot_fn, tot_tn, reduction="macro") 

    model.train()
    return {'dice_loss': dice_loss.mean()/max(num_val_batches, 1), 'per_image_iou': per_image_iou, 'dataset_iou': dataset_iou, 'dataset_sensitivity': dataset_sensitivity, 'dataset_specificity': dataset_specificity}
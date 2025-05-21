
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp

def calculate_metrics(logits_mask, gt_masks, num_classes=4):
    """Calculate F1 score and related metrics"""
    prob_mask = logits_mask.softmax(dim=1).argmax(dim=1)
    gt_masks = gt_masks.long()
    tp, fp, fn, tn = smp.metrics.get_stats(prob_mask, gt_masks, mode="multiclass", num_classes=num_classes)
    
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
    
    return {
        'f1': f1_score.mean().item(),
        'f2_score': f2_score.mean().item(),
        'precision': f2_score.mean().item(),
        'recall': recall.mean().item(),
        'iou': iou_score.mean().item(),
        'accuracy': accuracy.mean().item(),

    }

# class CombinedLoss(nn.Module):
#     def __init__(self, dice_weight=0.5, focal_weight=0.5):
#         super(CombinedLoss, self).__init__()
#         self.dice_weight = dice_weight
#         self.focal_weight = focal_weight
        
#         self.dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
#         self.focal_loss = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE)
        
#     def forward(self, outputs, targets):
#         dice = self.dice_loss(outputs, targets)
#         focal = self.focal_loss(outputs, targets)
#         return self.dice_weight * dice + self.focal_weight * focal


class JaccardDiceLoss(nn.Module):
    def __init__(self, jaccard_weight=0.5, dice_weight=0.5):
        super(JaccardDiceLoss, self).__init__()
        self.jaccard_weight = jaccard_weight
        self.dice_weight = dice_weight
        
        self.jaccard_loss = smp.losses.JaccardLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        
    def forward(self, outputs, targets):
        jaccard = self.jaccard_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        return self.jaccard_weight * jaccard + self.dice_weight * dice



def calculate_test_metrics(model, test_loader, device):
    """Calculate comprehensive metrics on test dataset"""
    #criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
    # criterion = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE)
    #criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
    # criterion = JaccardDiceLoss(jaccard_weight=0.5, dice_weight=0.5)
    criterion = smp.losses.JaccardLoss(smp.losses.MULTICLASS_MODE, from_logits=True, smooth=0.0, eps=1e-07)
    
    total_loss = 0
    test_metrics = {
        'f1': 0,
        'f2_score': 0,
        'precision': 0,
        'recall': 0,
        'iou': 0,
        'accuracy': 0
    }
    
    with torch.no_grad():
        for batch in test_loader:
            images, masks, metadata = batch  # Unpack metadata
                
            images = images.to(device)
            masks = masks.to(device)
            metadata = metadata.to(device)  # Move metadata to device
            
            # Apply the same preprocessing as in training
            masks = masks.argmax(dim=1)
            masks = masks.long()
            
            # Get model predictions with metadata
            outputs = model(images)
            outputs = outputs.contiguous()
            
            # Calculate loss
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Calculate batch metrics using the same function as in training
            batch_metrics = calculate_metrics(outputs, masks)
            
            # Accumulate metrics
            for k in test_metrics:
                test_metrics[k] += batch_metrics[k]
    
    # Calculate averages
    avg_loss = total_loss / len(test_loader)
    for k in test_metrics:
        test_metrics[k] /= len(test_loader)
    
    # Add loss to metrics dictionary
    test_metrics['test_loss'] = avg_loss
    
    return test_metrics
    
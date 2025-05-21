
import os
import json
from pathlib import Path


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
import segmentation_models_pytorch as smp
from metrics import calculate_metrics, calculate_test_metrics, JaccardDiceLoss
#CombinedLoss



import wandb

def train_model(model, train_loader, val_loader, device, num_epochs, learning_rate, wandbrun,patience=100, save_dir='.'):
    
    #criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
    # criterion = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE)
    #criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
    #criterion = smp.losses.JaccardLoss(smp.losses.MULTICLASS_MODE, from_logits=True, smooth=0.0, eps=1e-07)
    criterion = JaccardDiceLoss(jaccard_weight=0.5, dice_weight=0.5)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, threshold=0.01, min_lr=1e-6, verbose=True
    
    )
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    best_f1 = 0.0
    
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        train_metrics = {
                    'f1': 0,
                    'f2_score': 0,
                    'precision': 0,
                    'recall': 0,
                    'iou': 0,
                    'accuracy': 0
        }
          
        
        for batch_idx, batch in enumerate(train_loader):
            
        
            
            images, masks, metadata = batch
            images = images.to(device)
            masks = masks.to(device)
            masks = masks.argmax(dim=1)
            masks = masks.long()
            #metadata = metadata.to(device)
            
            
            # In the training loop
            # print(f'Input image shape: {images.shape}')
            # outputs = model(images)
            # print(f'Model output shape: {outputs.shape}')
            # outputs = outputs.contiguous() 
                       
            outputs = model(images)
            outputs = outputs.contiguous()
           
            
            loss = criterion(outputs, masks)
            
            batch_metrics = calculate_metrics(outputs, masks)
            
            for k in train_metrics:
                train_metrics[k] += batch_metrics[k]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 25 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
            
        # Calculate training averages
        train_loss = train_loss / len(train_loader)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = {
            'f1': 0,
            'f2_score': 0,
            'precision': 0,
            'recall': 0, 
            'iou': 0,
            'accuracy': 0   
        }
        
        with torch.no_grad():
            for batch in val_loader:
                images, masks, metadata = batch
                images = images.to(device)
                masks = masks.to(device)
                masks = masks.argmax(dim=1)
                masks = masks.long()
                metadata = metadata.to(device)
                

                outputs = model(images)
                outputs = outputs.contiguous()
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                batch_metrics = calculate_metrics(outputs, masks)
                for k in val_metrics:
                    val_metrics[k] += batch_metrics[k]
        
        # Calculate validation averages
        val_loss = val_loss / len(val_loader)
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Track current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        
        print(val_metrics)
        
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, F1: {train_metrics["f1"]:.4f}')
        print(f'Val Loss: {val_loss:.4f}, F1: {val_metrics["f1"]:.4f}')
        
        # Save best models
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_f1.pth'))
            print(f'Saved new best model with F1 score: {val_metrics["f1"]:.4f}')
            counter = 0  # Reset counter
        else:
            counter += 1
            
        if counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
            
        # Same for validation loss if you prefer to monitor that instead
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_loss.pth'))
            print(f'Saved new best model with validation loss: {val_loss:.4f}')
            # You can have separate counters for F1 and loss if you want
            
        # Log metrics to W&B
        wandbrun.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_f1': train_metrics['f1'],
            'val_f1': val_metrics['f1'],
            'train_precision': train_metrics['precision'],
            'val_precision': val_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'val_recall': val_metrics['recall'],
            'val_iou': val_metrics['iou'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f2_score': val_metrics['f2_score'],
            'learning_rate': current_lr
        })
        
    wandbrun.finish()
    
    return history

import os
import json
from pathlib import Path

from dataset import get_data_loaders
from metrics import calculate_metrics

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from PIL import Image
import segmentation_models_pytorch as smp

from model import get_model

import optuna 

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

def trainer(trial, device, data_params, params, paths, save_dir='.'):
    try:
        # Clear CUDA cache at the beginning of each trial
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Define loss functions
        focalloss = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE)
        jaccard_dice_loss = JaccardDiceLoss(jaccard_weight=0.5, dice_weight=0.5)
        
        
        # Use parameters passed from objective function
        if params is not None:
            encoder_name = params.get('encoder')
            batch_size = params.get('batch_size')
        else:
            # This should never happen with the new implementation
            raise ValueError("trial_params must be provided")

        # Still suggest other parameters that aren't in trial_params
        learning_rate = trial.suggest_float('learning_rate', 5e-6, 1e-2, log=True)
        criterion_sel = trial.suggest_categorical('criterion', ["focalloss", "jaccard_dice_loss"])

        if criterion_sel == "focalloss":
            criterion = focalloss
        elif criterion_sel == "jaccard_dice_loss":
            criterion = jaccard_dice_loss
        
        # No batch_size suggestion here - use the one from trial_params

        num_epochs = 100
        patience = 50
        augument = 'True'
        encoder_weights = 'imagenet'
        
        model = get_model(
            params=params,
            data_params=data_params,
        )

        model = model.to(device)

        # Get data loaders
        train_loader, val_loader, test_loader = get_data_loaders(
            base_dir=paths['dataset_base_path'],
            batch_size=batch_size,
            augment=augument,
            img_size=data_params['img_size'],
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
   
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            
        history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': []
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
                
                
            # Same for validation loss if you prefer to monitor that instead
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_loss.pth'))
                print(f'Saved new best model with validation loss: {val_loss:.4f}')
                # You can have separate counters for F1 and loss if you want
                
            # Log metrics to W&B
        #     wandbrun.log({
        #         'train_loss': train_loss,
        #         'val_loss': val_loss,
        #         'train_f1': train_metrics['f1'],
        #         'val_f1': val_metrics['f1'],
        #         'train_precision': train_metrics['precision'],
        #         'val_precision': val_metrics['precision'],
        #         'train_recall': train_metrics['recall'],
        #         'val_recall': val_metrics['recall'],
        #         'val_iou': val_metrics['iou'],
        #         'val_accuracy': val_metrics['accuracy'],
        #         'val_f2_score': val_metrics['f2_score']
        #     })
            
        # wandbrun.finish()

            trial.report(val_metrics['iou'], epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
            
        # Log final results and hyperparameters
        with open(os.path.join(save_dir, f'trial_{trial.number}_results.json'), 'w') as f:
            result = {
                'trial_number': trial.number,
                'hyperparameters': {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'patience': patience,
                    'use_augmentation': augument,
                    'encoder': encoder_name,
                    'encoder_weights': encoder_weights
                },
                'best_metrics': {
                    'iou': val_metrics['iou'],
                    'f1': best_f1,
                    'best_val_loss': best_val_loss
                },
                'history': history
            }
            json.dump(result, f, indent=4)

        print(f"Trial {trial.number} completed. Best IoU: {val_metrics['iou']:.4f}, Best F1: {best_f1:.4f}")
        return val_metrics['iou']
    
    except torch.cuda.OutOfMemoryError:
        # Handle GPU OOM error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache
        
        print(f"Trial {trial.number} - CUDA out of memory with parameters: batch_size={batch_size}, encoder={encoder_name}")
        # Return a very poor score so Optuna will avoid this region
        return float('-inf')  # or a very small value like -1.0
        
    except Exception as e:
        # Handle other exceptions
        print(f"Trial {trial.number} failed with error: {str(e)}")
        return float('-inf')

    






       

        
    
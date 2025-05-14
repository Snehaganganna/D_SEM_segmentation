

import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from model import UnetMeta  # Add this import
from model import get_model

def load_model(model_path, device, params ,data_params=None):
    """Load the trained model with weights_only=True"""
    model = get_model(params,data_params)
    
    # model = smp.Segformer(
    #     encoder_name=params['encoder'],
    #     encoder_weights="imagenet",
    #     in_channels=1,
    #     classes=4,
    #     activation=None,
    # )
    
    # model = smp.UnetPlusPlus(
    #     encoder_name=params['encoder'],
    #     encoder_weights="imagenet",
    #     in_channels=1,
    #     classes=4,
    #     activation=None,
    # )
    # model = UnetPlusPlusMeta(
    #     encoder_name=params['encoder'],
    #     encoder_weights="imagenet",
    #     in_channels=1,
    #     classes=4,
    #     activation=None,
    #     metadata_dim=4,  # This should match what you used during training
    #     fusion_type=params['fusion_type']  # Get fusion type from params
    # )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval() 
    return model

def visualize_prediction(image, mask, prediction, save_path=None):
    """Visualize predictions with class colors"""
    if torch.is_tensor(image):
        image = image.cpu().numpy().transpose(1, 2, 0)
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    
    # Define colors for each class
    colors = np.array([
        [0, 0, 255],      # Class 0 - Ductile - Blue
        [255, 255, 0],    # Class 1 - Brittle - Yellow
        [139, 69, 19],    # Class 2 - Background - Brown
        [0, 255, 0]       # Class 3 - Pores - Green
    ]) / 255.0
    
    # Create colored masks
    mask_display = np.zeros((*mask.shape[1:], 3))
    pred_display = np.zeros((*prediction.shape[1:], 3))
    
    mask_indices = np.argmax(mask, axis=0)
    pred_indices = np.argmax(prediction, axis=0)
    
    for class_idx, color in enumerate(colors):
        mask_display[mask_indices == class_idx] = color
        pred_display[pred_indices == class_idx] = color
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    axes[0].imshow(normalized_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask_display)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_display)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
import torch
import os
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import load_model
import yaml

def test_images(model_path, input_dir, output_dir, params, img_size=(960, 960)):
    """Test the model on new images without ground truth masks"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
    
    # Load model
    model = load_model(model_path, device, params)
    print(f"Model loaded from {model_path}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Define preprocessing
    mean = [0.449]
    std = [0.226]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Find image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        # Try both lowercase and uppercase extensions
        image_files.extend(list(Path(input_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(input_dir).glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    # Process each image
    successful_count = 0
    
    for i, img_path in enumerate(image_files):
        img_filename = os.path.basename(img_path)
        print(f"[{i+1}/{len(image_files)}] Processing {img_filename}")
        
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert('L')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                prediction = model(input_tensor)
            
            # Get class indices
            pred_softmax = prediction.softmax(dim=1)
            pred_indices = pred_softmax.argmax(dim=1).cpu().numpy().squeeze()
            
            # Save mask as numpy file
            mask_path = os.path.join(output_dir, "masks", f"{Path(img_filename).stem}_mask.npy")
            np.save(mask_path, pred_indices)
            print(f"  - Saved mask to {mask_path}")
            
            # Also save as PNG for easier viewing
            mask_png_path = os.path.join(output_dir, "masks", f"{Path(img_filename).stem}_mask.png")
            mask_img = Image.fromarray(pred_indices.astype(np.uint8))
            mask_img.save(mask_png_path)
            print(f"  - Saved mask PNG to {mask_png_path}")
            
            # Create visualization
            visualize_without_groundtruth(
                input_tensor[0], 
                pred_indices,
                save_path=os.path.join(output_dir, "visualizations", f"{Path(img_filename).stem}_viz.png")
            )
            print(f"  - Saved visualization")
            
            successful_count += 1
            
        except Exception as e:
            print(f"Error processing image {img_filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nTesting completed! Successfully processed {successful_count} out of {len(image_files)} images.")
    print(f"Results saved in {output_dir}")

def visualize_without_groundtruth(image, prediction, save_path=None):
    """Visualize prediction without ground truth comparison"""
    if torch.is_tensor(image):
        image = image.cpu().numpy().transpose(1, 2, 0)
    
    # Define colors for each class
    colors = np.array([
        [0, 0, 255],      # Class 0 - Ductile - Blue
        [255, 255, 0],    # Class 1 - Brittle - Yellow
        [139, 69, 19],    # Class 2 - Background - Brown
        [0, 255, 0]       # Class 3 - Pores - Green
    ]) / 255.0
    
    # Define class names
    class_names = ["Ductile", "Brittle", "Background", "Pores"]
    
    # Create colored mask
    colored_mask = np.zeros((*prediction.shape, 3))
    for class_idx, color in enumerate(colors):
        colored_mask[prediction == class_idx] = color
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Normalize image for display
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    if normalized_image.shape[0] == 1:  # Remove channel dimension if needed
        normalized_image = normalized_image.squeeze(0)
    
    # Plot original image
    ax1.imshow(normalized_image, cmap='gray')
    ax1.set_title('Original Image', fontsize=14)
    ax1.axis('off')
    
    # Plot prediction
    ax2.imshow(colored_mask)
    ax2.set_title('Segmentation Prediction', fontsize=14)
    ax2.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = []
    for i, (name, color) in enumerate(zip(class_names, colors)):
        # Only include classes that appear in the prediction
        if np.any(prediction == i):
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=name))
    
    if legend_elements:
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    # Load parameters from YAML file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    params = config['params']
    data_params = config['data_params']
    
    # Define paths
    model_path = r"D:\TEAM\Sneha\D_SEM_segmentation\output\unet_for_testing\best_model_f1.pth"
    input_dir = r"D:\TEAM\Sneha\D_SEM_segmentation\test_images"
    output_dir = r"D:\TEAM\Sneha\D_SEM_segmentation\test_results"
    
    # Test the model on new images
    test_images(
        model_path=model_path,
        input_dir=input_dir, 
        output_dir=output_dir,
        params=params,
        img_size=tuple(data_params['img_size'])
    )
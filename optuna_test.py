import torch
import os

from utils import load_model, visualize_prediction
from dataset import get_data_loaders
from metrics import calculate_test_metrics

import yaml

def test(params, data_params, paths, exp_folder):
    """Test the model and generate visualizations"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model from the experiment folder
    model_path = os.path.join(exp_folder, 'best_model_loss.pth')

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        print("Please run training first or specify the correct experiment folder.")
        return
    
    model = load_model(model_path, device, params, data_params)
    
    # Get data loaders (same as in training)
    train_loader, val_loader, test_loader = get_data_loaders(
        base_dir=paths['dataset_base_path'],
        batch_size=1,  # Use batch size 1 for testing
        augment=False,
        img_size=data_params['img_size'],
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create output directories
    predictions_dir = os.path.join(exp_folder, 'predictions')
    results_dir = os.path.join(exp_folder, 'results')
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Calculate metrics on test set
    print("Calculating metrics on test set...")
    metrics = calculate_test_metrics(model, test_loader, device)
        
    print(f"Test metrics: {metrics}")
    # Save metrics to a text file in the results directory
    metrics_file = os.path.join(results_dir, 'test_metrics.txt')
    with open(metrics_file, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    print(f"Test metrics saved to {metrics_file}")
    
    # Generate predictions
    print("\nGenerating visualizations...")
    with torch.no_grad():
        for i, (image, mask, metadata) in enumerate(test_loader):
            if i >= 100:  # Limit to first 100 predictions
                break
            
            try:
                image = image.to(device)
                metadata = metadata.to(device)
                prediction = model(image)
                
                save_path = os.path.join(predictions_dir, f'prediction_{i}.png')
                visualize_prediction(
                    image[0],
                    mask[0],
                    prediction[0],
                    save_path=save_path
                )
                print(f"Saved {save_path}")
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                continue
    
    print(f"Testing completed! Results saved in {exp_folder}")
    
if __name__ == '__main__':
    out_folder = "output" 
    exp_name = "dk_normal"
    
    exp_folder = os.path.join(out_folder, exp_name)
    
    # Load parameters from YAML file
    with open(os.path.join(exp_folder ,'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
        
    params = config['params']
    data_params = config['data_params']
    paths = config['paths']
    
    print(config)
    # Call the test function with both params and paths
    test(params,data_params, paths, exp_folder)
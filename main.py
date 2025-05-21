
import os
import segmentation_models_pytorch as smp
import torch
import json

from dataset import get_data_loaders
from plot import  plot_training_history
from trainer import train_model
import yaml

from model import get_model

import wandb

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
print(config)

params = config['params']
paths = config['paths']
data_params = config['data_params'] 

run = wandb.init(
    entity="gsneha61197-freiburg-university",
    project="sem-segmentation",
    config=config
)

output_folder = paths['output_folder']
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, 'predictions'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'results'), exist_ok=True)

with open(os.path.join(output_folder, 'config.yaml'), 'w') as yaml_file:
    yaml.dump(config, yaml_file)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# model = UnetMeta(
#         encoder_name=params['encoder'],
#         encoder_weights="imagenet",
#         in_channels=1,
#         classes=4,
#         activation=None,
#         metadata_dim=data_params['metadata_dim'],
#         fusion_type=params['fusion_type'],
#     )
    
# model = smp.Segformer(
#         encoder_name=params['encoder'],
#         encoder_weights="imagenet",
#         in_channels=1,
#         classes=4,
#         activation=None,
# )

# model = smp.Unet(
#         encoder_name=params['encoder'],
#         encoder_weights="imagenet",
#         in_channels=1,
#         classes=4,
#         activation=None,
    
#     )

model = get_model(params=params, data_params=data_params)
model = model.to(device)

# Get data loaders
train_loader, val_loader, test_loader = get_data_loaders(
    base_dir=paths['dataset_base_path'],
    batch_size=params['batch_size'],
    augment=data_params['augment'],
    img_size=data_params['img_size'],
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

# Train model
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=params['epochs'],
    patience=params['patience'],
    learning_rate=params['learning_rate'],
    save_dir=output_folder,
    wandbrun=run
)

# Plot and save training history
plot_training_history(history, save_dir=output_folder)

# Save training history
with open(os.path.join(output_folder, 'training_history.json'), 'w') as f:
    json.dump(history, f)

# Save hyperparameters
with open(os.path.join(output_folder, 'hyperparameters.json'), 'w') as f:
    json.dump(params, f)

print(f"Training completed! Results saved in {output_folder}")

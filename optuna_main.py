import yaml

import os
import segmentation_models_pytorch as smp
import torch
import json

from model import UnetMeta

import wandb

import optuna
from optuna_trainer import trainer
from optuna.trial import TrialState
from optuna.integration.wandb import WeightsAndBiasesCallback


# Load parameters from YAML file
with open('hyperopt.yaml', 'r') as file:
    config = yaml.safe_load(file)
    

params = config['params']
data_params = config['data_params']
paths = config['paths']

output_folder = paths['output_folder']

os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, 'predictions'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'results'), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open(os.path.join(output_folder, 'config.yaml'), 'w') as yaml_file:
    yaml.dump(config, yaml_file)

# Train model
def objective(trial):
    # Define encoder first
    encoder = trial.suggest_categorical('encoder', ['resnet18', 'resnet50'])

    # Now define batch size as a DIFFERENT parameter based on encoder
    if encoder == 'resnet18':
        batch_size = trial.suggest_categorical('batch_size_resnet18', [4, 8, 16])
    else:  # resnet50
        batch_size = trial.suggest_categorical('batch_size_resnet50', [4, 8])
    
    # Create a parameter dictionary to pass to trainer
    trial_params = {
        'encoder': encoder,
        'batch_size': batch_size,
        **params 
        }
    
    print(f"Trial parameters: {trial_params}")
    
    val_iou = trainer(
        trial,
        device=device,
        data_params=data_params,
        params=trial_params,
        paths=paths,
        save_dir=output_folder,
    )
    return val_iou


if __name__ == "__main__":
    wandb_kwargs = {"project": "sem-segmentation", "name":f"optuna_{paths['output_folder'].split('/')[-1]}"}
    print(f"wandb_kwargs: {wandb_kwargs}")
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=False)
    
    # Create study with custom pruner that's more tolerant of failed trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    
    try:
        study.optimize(objective, n_trials=100, callbacks=[wandbc])
    except KeyboardInterrupt:
        print("Optimization stopped by user.")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    failed_trials = study.get_trials(deepcopy=False, states=[TrialState.FAIL])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("  Number of failed trials: ", len(failed_trials))

    if len(complete_trials) > 0:
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("No trials completed successfully.")
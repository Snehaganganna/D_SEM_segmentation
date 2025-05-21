import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def print_model(model_type, metadata_dim=None):
    """
    Print information about model configuration.
    
    Args:
        model_type: 'Unet' or 'UnetMeta'
        metadata_dim: If provided, sets the metadata dimension for UnetMeta
    """
    print("\n" + "="*70)
    
    if model_type == 'Unet':
        print("MODEL: Unet")
        print("METADATA: NO")
        
        # Create standard Unet
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=1,
            classes=4,
            activation=None
        )
        
        # For standard Unet, metadata_dim is definitely 0
        has_metadata = False
        
    elif model_type == 'UnetMeta':
        print(f"MODEL: UnetMeta (metadata_dim={metadata_dim})")
        
        # Check if metadata is actually being used
        has_metadata = metadata_dim is not None and metadata_dim > 0
        
        if has_metadata:
            print("METADATA: YES")
        else:
            print("METADATA: NO (metadata_dim=0)")
        
        # Import your UnetMeta class
        try:
            from model import UnetMeta
            
            # Create UnetMeta
            model = UnetMeta(
                metadata_dim=metadata_dim if metadata_dim is not None else 0,
                fusion_type='concat',
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=1,
                classes=4,
                activation=None
            )
        except (ImportError, ModuleNotFoundError):
            print("ERROR: UnetMeta class not found")
            model = None
            
    else:
        print(f"MODEL: Unknown model type '{model_type}'")
        model = None
        has_metadata = False
    
    # Print parameter count if model was created successfully
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"PARAMETERS: {total_params:,}")
        
        # Print architecture
        print("\nARCHITECTURE:")
        print("-" * 70)
        
        # Process the model's string representation for better readability
        model_str = str(model)
        lines = model_str.split('\n')
        
        for line in lines:
            # Calculate indentation level
            indent_level = len(line) - len(line.lstrip())
            clean_line = line.strip()
            
            # Skip empty lines
            if not clean_line:
                continue
                
            # Highlight metadata-related components
            if has_metadata and any(x in clean_line.lower() for x in ['metadata', 'bottleneck_adapter']):
                print(f"{' ' * indent_level}➡️ {clean_line}")
            else:
                print(f"{' ' * indent_level}{clean_line}")
    
    print("="*70)


# Example usage
if __name__ == "__main__":
    # Print standard Unet
    print_model('Unet')
    
    # Print UnetMeta with metadata_dim=4
    print_model('UnetMeta', metadata_dim=4)
    
    # Print UnetMeta with metadata_dim=0
    print_model('UnetMeta', metadata_dim=0)
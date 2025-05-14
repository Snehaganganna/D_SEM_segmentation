import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


  
class UnetMeta(smp.Unet):
    def __init__(self, metadata_dim, fusion_type="concat", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata_dim = metadata_dim
        self.fusion_type = fusion_type
        
        # Add metadata processing components if metadata dimension is provided
        if metadata_dim > 0:
            # Feature dimension at bottleneck (deepest encoder feature)
            bottleneck_dim = self.encoder.out_channels[-1]
            
            # Create metadata projection network
            self.metadata_projection = nn.Sequential(
                nn.Linear(metadata_dim, 256),
                nn.ReLU(),
                nn.Linear(256, bottleneck_dim),
                nn.ReLU()
            )
            
            # For concatenation, we need to adjust decoder input dimension
            if fusion_type == 'concat':
                # Modify first decoder block to accept concatenated features
                self.bottleneck_adapter = nn.Conv2d(
                    bottleneck_dim * 2,  # Doubled due to concatenation
                    bottleneck_dim,      # Back to original dimension
                    kernel_size=1        # 1x1 convolution
                )
                
    def forward(self, x, metadata=None):
        """Sequentially pass `x` through model's encoder, decoder and heads
        
        Args:
            x: Input image tensor
            metadata: Optional metadata tensor of shape [B, metadata_dim]
        """
        #print(f"Forward pass - input shape: {x.shape}")
        
        #print("Starting encoder...")
        features = self.encoder(x)
        #print("Encoder complete")
        
        # Process metadata if provided
        if metadata is not None and self.metadata_dim > 0:
            # Project metadata to match feature dimensions
            batch_size = x.shape[0]
            metadata_features = self.metadata_projection(metadata)
            
            # Reshape for spatial integration
            # [B, C] -> [B, C, 1, 1] to match spatial dimensions of bottleneck features
            metadata_features = metadata_features.view(batch_size, -1, 1, 1)
            
            # Integrate metadata with bottleneck features based on fusion type
            if self.fusion_type == 'concat':
                # Expand metadata features to match spatial dimensions of bottleneck
                expanded_metadata = metadata_features.expand(
                    -1, -1, features[-1].size(2), features[-1].size(3)
                )
                
                # Concatenate along channel dimension
                concatenated = torch.cat([features[-1], expanded_metadata], dim=1)
                
                # Apply 1x1 convolution to restore original channel dimension
                features[-1] = self.bottleneck_adapter(concatenated)
                
            elif self.fusion_type == 'add':
                # Element-wise addition
                features[-1] = features[-1] + metadata_features
                
            elif self.fusion_type == 'multiply':
                # Element-wise multiplication
                features[-1] = features[-1] * metadata_features
        
        decoder_output = self.decoder(features)
        
        masks = self.segmentation_head(decoder_output)


        return masks
    


class DeepLabV3PlusMeta(smp.DeepLabV3Plus):
    """
    DeepLabV3+ model with metadata integration at the bottleneck
    """
    def __init__(self, metadata_dim, fusion_type="concat", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata_dim = metadata_dim
        self.fusion_type = fusion_type
        
        # Add metadata processing components if metadata dimension is provided
        if metadata_dim > 0:
            # Feature dimension at the ASPP module output
            # In DeepLabV3+, this corresponds to the bottleneck features
            aspp_out_channels = self.decoder.aspp.out_channels
            
            # Create metadata projection network
            self.metadata_projection = nn.Sequential(
                nn.Linear(metadata_dim, 256),
                nn.ReLU(),
                nn.Linear(256, aspp_out_channels),
                nn.ReLU()
            )
            
            # For concatenation, we need to adjust decoder input dimension
            if fusion_type == 'concat':
                # The fusion adapter is applied after ASPP module but before decoder blocks
                self.bottleneck_adapter = nn.Conv2d(
                    aspp_out_channels * 2,  # Doubled due to concatenation
                    aspp_out_channels,      # Back to original dimension
                    kernel_size=1           # 1x1 convolution
                )
                
    def forward(self, x, metadata=None):
        """Sequentially pass `x` through model's encoder, ASPP, decoder and heads
        
        Args:
            x: Input image tensor
            metadata: Optional metadata tensor of shape [B, metadata_dim]
        """
        # Get features from encoder
        features = self.encoder(x)
        
        # Original DeepLabV3+ passes the high-res feature and the output from ASPP to the decoder
        # We'll modify the ASPP output with metadata before passing it to the decoder
        aspp_input = features[-1]
        
        # Process high-res features from the encoder
        high_res_features = self.decoder.skip_connection(features[2])
        
        # Process input through ASPP module
        aspp_output = self.decoder.aspp(aspp_input)
        
        # Process metadata if provided
        if metadata is not None and self.metadata_dim > 0:
            # Project metadata to match feature dimensions
            batch_size = x.shape[0]
            metadata_features = self.metadata_projection(metadata)
            
            # Reshape for spatial integration
            # [B, C] -> [B, C, 1, 1] to match spatial dimensions of ASPP output
            metadata_features = metadata_features.view(batch_size, -1, 1, 1)
            
            # Expand metadata features to match spatial dimensions of ASPP output
            h, w = aspp_output.size(2), aspp_output.size(3)
            expanded_metadata = metadata_features.expand(-1, -1, h, w)
            
            # Integrate metadata with ASPP output based on fusion type
            if self.fusion_type == 'concat':
                # Concatenate along channel dimension
                concatenated = torch.cat([aspp_output, expanded_metadata], dim=1)
                
                # Apply 1x1 convolution to restore original channel dimension
                aspp_output = self.bottleneck_adapter(concatenated)
                
            elif self.fusion_type == 'add':
                # Element-wise addition
                aspp_output = aspp_output + expanded_metadata
                
            elif self.fusion_type == 'multiply':
                # Element-wise multiplication
                aspp_output = aspp_output * expanded_metadata
        
        # Continue with the rest of DeepLabV3+ forward pass
        # But now with metadata-enhanced ASPP output
        x = self.decoder.project_aspp(aspp_output)
        
        # Upsample ASPP output to match high-res features
        x = nn.functional.interpolate(
            x, size=high_res_features.shape[-2:], 
            mode="bilinear", align_corners=True
        )
        
        # Concatenate with high-res features and process
        x = torch.cat([x, high_res_features], dim=1)
        x = self.decoder.projection_head(x)
        
        # Apply segmentation head
        x = self.segmentation_head(x)
        
        return x





def get_model (params, data_params=None):
    """Get the model based on the parameters"""
    model_name = params['model_name']
    print(f"Model name: {model_name}")
    if model_name == 'UnetMeta':
        # Use the custom UnetMeta model
        model = UnetMeta(
            metadata_dim=data_params['metadata_dim'],
            fusion_type=params['fusion_type'],
            encoder_name=params['encoder'],
            encoder_weights="imagenet",
            in_channels=1,
            classes=4,
            activation=None
        )
    elif model_name == 'Unet':
        # Use the default Unet model from segmentation_models_pytorch
        model = smp.Unet(
            encoder_name=params['encoder'],
            encoder_weights="imagenet",
            in_channels=1,
            classes=4,
            activation=None,
        )
        
    elif model_name == 'UnetPlusPlus':
        # Use the UnetPlusPlus model from segmentation_models_pytorch
        model = smp.UnetPlusPlus(
            encoder_name=params['encoder'],
            encoder_weights="imagenet",
            in_channels=1,
            classes=4,
            activation=None,
        )
   
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    return model    
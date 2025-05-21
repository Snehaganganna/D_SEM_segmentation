import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, metadata_dim):
        super().__init__()
        self.film_gen = nn.Sequential(
            nn.Linear(metadata_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim * 2)  # Generate gamma and beta
        )
        
    def forward(self, x, metadata):
        film_params = self.film_gen(metadata)
        gamma, beta = torch.split(film_params, x.size(1), dim=1)
        
        # Reshape for multiplication and addition
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta = beta.view(-1, x.size(1), 1, 1)
        
        # Apply FiLM conditioning
        return gamma * x + beta

class UnetMetaFiLM(smp.Unet):
    def __init__(self, metadata_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata_dim = metadata_dim
        
        if metadata_dim > 0:
            # Create FiLM layers for bottleneck and decoder stages
            bottleneck_dim = self.encoder.out_channels[-1]
            self.bottleneck_film = FiLMLayer(bottleneck_dim, metadata_dim)
            
    def forward(self, x, metadata=None):
        features = self.encoder(x)
        
        # Apply FiLM to bottleneck
        if metadata is not None and self.metadata_dim > 0:
            features[-1] = self.bottleneck_film(features[-1], metadata)
        
        decoder_output = self.decoder(features)
        masks = self.segmentation_head(decoder_output)
        
        return masks


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
    and additional upsampling to match input resolution
    """
    def __init__(self, metadata_dim, fusion_type="concat", upscale_output=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata_dim = metadata_dim
        self.fusion_type = fusion_type
        self.upscale_output = upscale_output
        
        # Add metadata processing components if metadata dimension is provided
        if metadata_dim > 0:
            # In newer versions of SMP, the decoder structure might be different
            # We need to determine the ASPP output channels dynamically
            
            # Check decoder structure to find ASPP output channels
            if hasattr(self.decoder, 'project'):
                # Find the input channels to the project layer
                aspp_out_channels = self.decoder.project.in_channels
            elif hasattr(self.decoder, 'aspp') and hasattr(self.decoder.aspp, 'project'):
                # Alternative structure - ASPP followed by project
                aspp_out_channels = self.decoder.aspp.project.out_channels
            elif isinstance(self.decoder, nn.Sequential) and len(self.decoder) > 0:
                # If decoder is a Sequential, try to find the first Conv layer
                for module in self.decoder.modules():
                    if isinstance(module, nn.Conv2d):
                        aspp_out_channels = module.in_channels
                        break
                else:
                    # Fallback to a reasonable default if we can't determine
                    print("Warning: Could not determine ASPP output channels, using default of 256")
                    aspp_out_channels = 256
            else:
                # Fallback to a reasonable default
                print("Warning: Could not determine ASPP output channels, using default of 256")
                aspp_out_channels = 256
            
            print(f"Using ASPP output channels: {aspp_out_channels}")
            
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
        """
        Forward pass with metadata integration and full-resolution output
        
        Args:
            x: Input image tensor
            metadata: Optional metadata tensor of shape [B, metadata_dim]
        """
        # Remember input size for later upsampling
        input_size = x.shape[-2:]
        
        # Extract features from the encoder
        features = self.encoder(x)
        
        # Need to adapt our approach based on the actual DeepLabV3Plus implementation
        # First, let's determine the structure by inspecting the decoder
        
        # For newer versions of SMP with Sequential decoder
        if isinstance(self.decoder, nn.Sequential):
            # We need to manually recreate the forward pass with metadata integration
            
            # Get the deepest encoder features (equivalent to aspp_input)
            encoder_output = features[-1]
            
            # Check if we have high-res features from skip connection
            # Typically the third feature map in ResNet-based encoders
            skip_idx = 2 if len(features) > 2 else 0
            high_res_features = features[skip_idx] if skip_idx < len(features) else None
            
            # Process through initial modules (which should include ASPP)
            # We'll need to determine the splitting point for metadata integration
            
            # For simplicity, let's assume the first module is ASPP or equivalent
            # and we'll integrate metadata after that
            if len(self.decoder) > 0:
                # Process through first module (ASPP or equivalent)
                aspp_output = self.decoder[0](encoder_output)
                
                # Process metadata if provided
                if metadata is not None and self.metadata_dim > 0:
                    # Project metadata to match feature dimensions
                    batch_size = x.shape[0]
                    metadata_features = self.metadata_projection(metadata)
                    
                    # Reshape for spatial integration
                    metadata_features = metadata_features.view(batch_size, -1, 1, 1)
                    
                    # Expand metadata features to match spatial dimensions of ASPP output
                    h, w = aspp_output.size(2), aspp_output.size(3)
                    expanded_metadata = metadata_features.expand(-1, -1, h, w)
                    
                    # Apply fusion strategy
                    if self.fusion_type == 'concat':
                        concatenated = torch.cat([aspp_output, expanded_metadata], dim=1)
                        aspp_output = self.bottleneck_adapter(concatenated)
                    elif self.fusion_type == 'add':
                        aspp_output = aspp_output + expanded_metadata
                    elif self.fusion_type == 'multiply':
                        aspp_output = aspp_output * expanded_metadata
                
                # Process through remaining decoder modules
                x = aspp_output
                for i in range(1, len(self.decoder)):
                    if i == 1 and high_res_features is not None:
                        # This is where the skip connection is typically used
                        # First upsampling then concatenation with high-res features
                        x = nn.functional.interpolate(
                            x, size=high_res_features.shape[-2:],
                            mode='bilinear', align_corners=True
                        )
                        x = torch.cat([x, high_res_features], dim=1)
                    
                    x = self.decoder[i](x)
            else:
                x = encoder_output
            
            # Apply segmentation head
            x = self.segmentation_head(x)
            
            # Upscale output to input resolution if requested
            if self.upscale_output:
                x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
            
            return x
        
        # For older versions with structured decoder
        else:
            # Original DeepLabV3+ approach
            aspp_input = features[-1]
            
            # Get high-res features for skip connection
            skip_features = self.decoder.skip_connection(features[2]) if hasattr(self.decoder, 'skip_connection') else None
            
            # Process input through ASPP
            aspp_output = self.decoder.aspp(aspp_input) if hasattr(self.decoder, 'aspp') else aspp_input
            
            # Process metadata if provided
            if metadata is not None and self.metadata_dim > 0:
                # Project metadata to match feature dimensions
                batch_size = x.shape[0]
                metadata_features = self.metadata_projection(metadata)
                
                # Reshape for spatial integration
                metadata_features = metadata_features.view(batch_size, -1, 1, 1)
                
                # Expand metadata features to match spatial dimensions of ASPP output
                h, w = aspp_output.size(2), aspp_output.size(3)
                expanded_metadata = metadata_features.expand(-1, -1, h, w)
                
                # Apply fusion strategy
                if self.fusion_type == 'concat':
                    concatenated = torch.cat([aspp_output, expanded_metadata], dim=1)
                    aspp_output = self.bottleneck_adapter(concatenated)
                elif self.fusion_type == 'add':
                    aspp_output = aspp_output + expanded_metadata
                elif self.fusion_type == 'multiply':
                    aspp_output = aspp_output * expanded_metadata
            
            # Continue with remaining decoder operations
            if hasattr(self.decoder, 'project_aspp'):
                x = self.decoder.project_aspp(aspp_output)
            else:
                x = aspp_output
            
            # Handle skip connection if available
            if skip_features is not None:
                # Upsample ASPP output to match high-res features
                x = nn.functional.interpolate(
                    x, size=skip_features.shape[-2:],
                    mode="bilinear", align_corners=True
                )
                
                # Concatenate with high-res features
                x = torch.cat([x, skip_features], dim=1)
                
                # Process through projection head if available
                if hasattr(self.decoder, 'projection_head'):
                    x = self.decoder.projection_head(x)
            
            # Apply segmentation head
            x = self.segmentation_head(x)
            
            # Upscale output to input resolution if requested
            if self.upscale_output:
                x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
            
            return x


# Updated get_model function to include the upscale_output parameter
def get_model(params, data_params=None):
    """
    Get the model based on the parameters
    
    Args:
        params: Dictionary containing model parameters
        data_params: Dictionary containing data parameters including metadata dimensions
        
    Returns:
        Initialized model
    """
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
    
    # In your get_model function
    elif model_name == 'UnetMetaFiLM':
        # Use the custom UnetMetaFiLM model
        model = UnetMetaFiLM(
            metadata_dim=data_params['metadata_dim'],
            encoder_name=params['encoder'],
            encoder_weights="imagenet",
            in_channels=1,
            classes=4,
            activation=None
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
    
    elif model_name == 'DeepLabV3Plus':
        # Use the DeepLabV3Plus model from segmentation_models_pytorch
        model = smp.DeepLabV3Plus(
            encoder_name=params['encoder'],
            encoder_weights="imagenet",
            in_channels=1,
            classes=4,
            activation=None,
        )
    
    elif model_name == 'DeepLabV3PlusMeta':
        # Use the custom DeepLabV3PlusMeta model with metadata integration
        # Set upscale_output=True to get full resolution output
        model = DeepLabV3PlusMeta(
            metadata_dim=data_params['metadata_dim'],
            fusion_type=params['fusion_type'],
            upscale_output=True,  # This is the key parameter to get full resolution
            encoder_name=params['encoder'],
            encoder_weights="imagenet",
            in_channels=1,
            classes=4,
            activation=None
        )
    
    elif model_name == 'SegFormer':
        # Use the SegFormer model from segmentation_models_pytorch
        model = smp.Segformer(
            encoder_name=params.get('encoder', 'mit_b0'),  # Default to mit_b0 if not specified
            encoder_weights="imagenet",
            in_channels=1,
            classes=4,
            activation=None,
        )
    
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    return model
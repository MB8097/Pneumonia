import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Dict, Optional, Tuple
from config import DEFAULT_DATA_CONFIG
# ACTIVATION FUNCTIONS
def get_activation(name: str) -> nn.Module:
    """Get activation function by name"""
    activations = {
        'relu': nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
        'elu': nn.ELU(inplace=True),
        'selu': nn.SELU(inplace=True),
        'gelu': nn.GELU(),
        'mish': nn.Mish(inplace=True),
        'swish': nn.SiLU(inplace=True),
        'tanh': nn.Tanh(),
        'prelu': nn.PReLU(),
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    
    return activations[name]

# CUSTOM CNN ARCHITECTURE
class ConvBlock(nn.Module):
    """
    Convolutional block with optional batch normalization and dropout
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        pool: bool = True
    ):
        super(ConvBlock, self).__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        ]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(get_activation(activation))
        
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        activation: str = 'relu',
        channels: List[int] = [32, 64, 128, 256],
        fc_sizes: List[int] = [512, 256],
        dropout_conv: float = 0.0,
        dropout_fc: float = 0.5,
        use_batch_norm: bool = True,
        img_size: int = 150
    ):
        super(CustomCNN, self).__init__()
        self.config = {
            'activation': activation,
            'channels': channels,
            'fc_sizes': fc_sizes,
            'dropout_conv': dropout_conv,
            'dropout_fc': dropout_fc,
            'use_batch_norm': use_batch_norm
        }
        # Build convolutional layers
        conv_layers = []
        curr_channels = in_channels
        for out_channels in channels:
            conv_layers.append(
                ConvBlock(
                    curr_channels, 
                    out_channels,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                    dropout=dropout_conv
                )
            )
            curr_channels = out_channels
        self.conv_layers = nn.Sequential(*conv_layers)
        # Calculate flattened size
        self._to_linear = self._get_conv_output_size(in_channels, img_size)
        # Build fully connected layers
        fc_layers = []
        fc_input = self._to_linear
        for fc_size in fc_sizes:
            fc_layers.extend([
                nn.Linear(fc_input, fc_size),
                get_activation(activation),
                nn.Dropout(dropout_fc)
            ])
            fc_input = fc_size
        # Output layer
        fc_layers.append(nn.Linear(fc_input, num_classes))
        if num_classes == 1:
            fc_layers.append(nn.Sigmoid())
        
        self.fc_layers = nn.Sequential(*fc_layers)
        # Initialize weights
        self._initialize_weights()
    def _get_conv_output_size(self, in_channels: int, img_size: int) -> int:
        """Calculate the size of flattened features after conv layers"""
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            dummy = self.conv_layers(dummy)
            return dummy.view(1, -1).size(1)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# RESIDUAL NETWORK
class ResidualBlock(nn.Module):
    """Basic residual block"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = 'relu',
        use_batch_norm: bool = True
    ):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.act1 = get_activation(activation)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.act2 = get_activation(activation)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.act2(out)
        return out


class CustomResNet(nn.Module):
    """
    Custom ResNet implementation for ablation studies
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        activation: str = 'relu',
        channels: List[int] = [64, 128, 256, 512],
        blocks_per_stage: List[int] = [2, 2, 2, 2],
        dropout: float = 0.5,
        use_batch_norm: bool = True
    ):
        super(CustomResNet, self).__init__()
        
        self.config = {
            'activation': activation,
            'channels': channels,
            'blocks_per_stage': blocks_per_stage,
            'dropout': dropout
        }
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]) if use_batch_norm else nn.Identity(),
            get_activation(activation),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        # Residual stages
        self.stages = nn.ModuleList()
        curr_channels = channels[0]
        for i, (out_channels, num_blocks) in enumerate(zip(channels, blocks_per_stage)):
            stride = 1 if i == 0 else 2
            stage = self._make_stage(curr_channels, out_channels, num_blocks, stride, activation, use_batch_norm)
            self.stages.append(stage)
            curr_channels = out_channels
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Identity()
        )
    def _make_stage(self, in_channels, out_channels, num_blocks, stride, activation, use_batch_norm):
        layers = [ResidualBlock(in_channels, out_channels, stride, activation, use_batch_norm)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, activation, use_batch_norm))
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for stage in self.stages:
            x = stage(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



# PRETRAINED MODELS
class PretrainedModel(nn.Module):
    def __init__(
        self,
        model_name: str = 'resnet18',
        pretrained: bool = True,
        num_classes: int = 1,
        in_channels: int = 1,
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        super(PretrainedModel, self).__init__()
        
        self.model_name = model_name
        self.config = {
            'model_name': model_name,
            'pretrained': pretrained,
            'dropout': dropout,
            'freeze_backbone': freeze_backbone
        }
        # Load pretrained model
        if model_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            self._modify_first_conv(in_channels)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = self._make_classifier(num_features, num_classes, dropout)
            
        elif model_name == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            self._modify_first_conv(in_channels)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = self._make_classifier(num_features, num_classes, dropout)
            
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self._modify_first_conv(in_channels)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = self._make_classifier(num_features, num_classes, dropout)
            
        elif model_name == 'densenet121':
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            self._modify_densenet_first_conv(in_channels)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = self._make_classifier(num_features, num_classes, dropout)
            
        elif model_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            self._modify_efficientnet_first_conv(in_channels)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._make_classifier(num_features, num_classes, dropout)
            
        elif model_name == 'vgg16':
            weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vgg16(weights=weights)
            self._modify_vgg_first_conv(in_channels)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(num_features, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes),
                nn.Sigmoid() if num_classes == 1 else nn.Identity()
            )
            
        elif model_name == 'mobilenet_v3':
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            self._modify_mobilenet_first_conv(in_channels)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = self._make_classifier(num_features, num_classes, dropout)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _make_classifier(self, in_features: int, num_classes: int, dropout: float) -> nn.Module:
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        ]
        if num_classes == 1:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
    
    def _modify_first_conv(self, in_channels: int):
        """Modify first conv layer for ResNet models"""
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
    
    def _modify_densenet_first_conv(self, in_channels: int):
        """Modify first conv layer for DenseNet"""
        if in_channels != 3:
            self.backbone.features.conv0 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
    
    def _modify_efficientnet_first_conv(self, in_channels: int):
        """Modify first conv layer for EfficientNet"""
        if in_channels != 3:
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
    
    def _modify_vgg_first_conv(self, in_channels: int):
        """Modify first conv layer for VGG"""
        if in_channels != 3:
            self.backbone.features[0] = nn.Conv2d(
                in_channels, 64, kernel_size=3, padding=1
            )
    
    def _modify_mobilenet_first_conv(self, in_channels: int):
        """Modify first conv layer for MobileNet"""
        if in_channels != 3:
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )
    
    def _freeze_backbone(self):
        """Freeze backbone parameters (for transfer learning)"""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)



# MODEL FACTORY
class ModelFactory:
    """Factory class to create models with different configurations"""
    
    @staticmethod
    def create(
        model_type: str = 'custom_cnn',
        **kwargs
    ) -> nn.Module:
        
        if model_type == 'custom_cnn':
            return CustomCNN(**kwargs)
        
        elif model_type == 'custom_resnet':
            return CustomResNet(**kwargs)
        
        elif model_type == 'pretrained':
            return PretrainedModel(**kwargs)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_summary(model: nn.Module) -> Dict:
        """Get model summary (parameters, layers, etc.)"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params
        }



# TESTING
if __name__ == "__main__":
    print("Testing model architectures...")
    # Test input
    x = torch.randn(2, 1, 150, 150)
    # Test CustomCNN
    print("\n1. Testing CustomCNN...")
    model = CustomCNN(activation='relu', channels=[32, 64, 128], dropout_fc=0.5)
    out = model(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    summary = ModelFactory.get_model_summary(model)
    print(f"   Parameters: {summary['total_params']:,}")
    # Test CustomResNet
    print("\n2. Testing CustomResNet...")
    model = CustomResNet(channels=[32, 64, 128, 256], blocks_per_stage=[1, 1, 1, 1])
    out = model(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    summary = ModelFactory.get_model_summary(model)
    print(f"   Parameters: {summary['total_params']:,}")
    # Test PretrainedModel
    print("\n3. Testing PretrainedModel (ResNet18)...")
    model = PretrainedModel(model_name='resnet18', pretrained=False, in_channels=1)
    out = model(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    summary = ModelFactory.get_model_summary(model)
    print(f"   Parameters: {summary['total_params']:,}")
    # Test different activations
    print("\n4. Testing different activations...")
    for act in ['relu', 'leaky_relu', 'elu', 'gelu', 'mish']:
        model = CustomCNN(activation=act, channels=[32, 64])
        out = model(x)
        print(f"   {act}: Output shape {out.shape}")
    print("\nAll tests passed!")
import torch.nn as nn
import torch
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Improved2DCNN(nn.Module):
    def __init__(self, dropout_rate=0.3, num_classes=15):
        super().__init__()
        
        # Feature Extractor: 4 Blocks with Stacked Convolutions
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5),
            
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5),
            
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5),

            # Block 4: 4x4 -> 2x2
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.GELU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Adaptive Pooling: Force feature maps to (1, 1) regardless of input resolution
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier Head (512 inputs instead of 4096)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), 
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def make_efficientnet(dropout_rate=0.2, num_classes=100):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, 
                                     stride=1, padding=1, bias=False)
    
    # Overwrite the default PyTorch Dropout layer (model.classifier[0])
    model.classifier[0] = nn.Dropout(p=dropout_rate, inplace=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model.to(device)


class LogisticRegressionBaseline(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.linear = nn.Linear(3 * 32 * 32, num_classes)    # flatten the image straight to class logits

    def forward(self, x):
        return self.linear(torch.flatten(x, 1))
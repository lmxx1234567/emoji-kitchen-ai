import torch
import torch.nn as nn
import torchvision.models as models

class SharedFeatureExtractor(nn.Module):
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        super().__init__()
        # Feature extraction with pretrained model (e.g., ResNet50)
        self.feature_extractor = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])

    def forward(self, x):
        return self.feature_extractor(x)
    
    def set_grad_requires(self, requires_grad):
        for param in self.feature_extractor.parameters():
            param.requires_grad = requires_grad

class ImageMerger(nn.Module):
    def __init__(self):
        super(ImageMerger, self).__init__()

        self.feature_extractor = SharedFeatureExtractor.get_instance()
        
        # Transposed convolution layers
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(512)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(256)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(128)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(64)
        )
        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1,bias=False)

    def forward(self, img1, img2):
        x1 = self.feature_extractor(img1)
        x2 = self.feature_extractor(img2)

        # Flatten and concatenate
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        
        # Pass through custom head
        x = x.view(-1, 1024, 4, 4)  # Reshape for transposed convolutions
        
        # Transposed convolutions
        x = nn.ReLU()(self.deconv1(x))
        x = nn.ReLU()(self.deconv2(x))
        x = nn.ReLU()(self.deconv3(x))
        x = nn.ReLU()(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))
        
        return x

# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.feature_extractor = SharedFeatureExtractor.get_instance()
        
        # Define the final layers for combining features and producing a decision
        # Assume the output feature map size is 512x4x4 for each image from ResNet
        self.final_layers = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 3, 64), # Concatenating three feature maps along channel dimension
            nn.BatchNorm1d(64),  # BatchNorm1d for fully connected layers
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img1, img2, merged_img):
        # Process each image through the encoder
        x1 = self.feature_extractor(img1)
        x2 = self.feature_extractor(img2)
        x_merged = self.feature_extractor(merged_img)
        
        # Combine the feature maps (concatenating them along the channel dimension)
        x_combined = torch.cat((x1, x2, x_merged), dim=1)
        
        # Flatten the combined feature maps
        x_combined = x_combined.view(x_combined.size(0), -1)
        
        # Pass through the final layers to produce a decision
        decision = self.final_layers(x_combined)
        
        return decision
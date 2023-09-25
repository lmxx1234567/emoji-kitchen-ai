import torch
import torch.nn as nn
import torchvision.models as models

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Load a pre-trained ResNet model, removing the final fully connected layer
        self.resnet_encoder = models.resnet34(pretrained=True)
        self.resnet_encoder = nn.Sequential(*list(self.resnet_encoder.children())[:-2])
        
        # Define the final layers for combining features and producing a decision
        # Assume the output feature map size is 512x4x4 for each image from ResNet
        self.final_layers = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 3, 64),  # Concatenating three feature maps along channel dimension
            # dropout layer
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img1, img2, merged_img):
        # Process each image through the encoder
        x1 = self.resnet_encoder(img1)
        x2 = self.resnet_encoder(img2)
        x_merged = self.resnet_encoder(merged_img)
        
        # Combine the feature maps (concatenating them along the channel dimension)
        x_combined = torch.cat((x1, x2, x_merged), dim=1)
        
        # Flatten the combined feature maps
        x_combined = x_combined.view(x_combined.size(0), -1)
        
        # Pass through the final layers to produce a decision
        decision = self.final_layers(x_combined)
        
        return decision


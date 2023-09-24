import torch
import torch.nn as nn
import torchvision.models as models

class ImageMerger(nn.Module):
    def __init__(self):
        super(ImageMerger, self).__init__()
        
        # Feature extraction with pretrained model (e.g., ResNet50)
        self.feature_extractor = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        
        # Transposed convolution layers
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

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
    
    def set_feature_extractor_grad(self, requires_grad):
        for param in self.feature_extractor.parameters():
            param.requires_grad = requires_grad

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = ImageMerger().to(device)  # Changed 'mps_device' to the more generic 'device'
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

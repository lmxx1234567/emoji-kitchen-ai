import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class ImageMerger(nn.Module):
    def __init__(self):
        super(ImageMerger, self).__init__()
        
        # Feature extraction with pretrained model (e.g., ResNet50)
        self.feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        
        # Custom head for merging
        self.fc1 = nn.Linear(32768 * 2, 1024)  # 2048 * 4 * 4
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128 * 128 * 3)  # assuming the output is a 128x128 RGB image

    def forward(self, img1, img2):
        x1 = self.feature_extractor(img1)
        x2 = self.feature_extractor(img2)

        # Flatten and concatenate
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        
        # Pass through custom head
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x.view(-1, 3, 128, 128)  # reshaping to the desired output shape
    
# if __name__ == "__main__":
#     mps_device = torch.device("mps")
#     model = ImageMerger().to(mps_device)  # 'device' can be 'cuda' or 'cpu'
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
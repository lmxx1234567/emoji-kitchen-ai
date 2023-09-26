from torch.utils.data import DataLoader
from dataset import EmojiDataset
import torch
import torch.nn as nn
import torch.optim as optim
from models import ImageMerger
from tqdm import tqdm
import os
import argparse

def initialize_weights(layer):
    if isinstance(layer, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


def train(checkpoint_path=None,frezze_feature_extractor=False):
    # 'device' can be 'cuda', 'mps' or 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    model = ImageMerger().to(device)

    # Freeze feature extractor
    if frezze_feature_extractor:
        model.set_feature_extractor_grad(False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # load generator
    model,optimizer,start_epoch = load_generator(checkpoint_path=checkpoint_path,model=model,optimizer=optimizer)

    dataset = EmojiDataset("data/emojiData.json")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Define hyperparameters
    num_epochs = 25
    print_every = 100  # Print training stats every 100 batches

    for epoch in range(start_epoch, num_epochs):
        # Init progess bar
        pbar = tqdm(total=len(dataloader))
        running_loss = 0.0
        for i, (left_images, right_images, target_images) in enumerate(dataloader):
            # Move data to device
            left_images, right_images, target_images = (
                left_images.to(device),
                right_images.to(device),
                target_images.to(device),
            )

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(left_images, right_images)

            # Compute loss
            loss = criterion(outputs, target_images)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % print_every == 0:
                running_loss = 0.0
            else:
                pbar.set_postfix(loss=running_loss / (i%print_every))
            pbar.update(1)
            pbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
           
        # Save model checkpoint
        save_generator(checkpoint_path=checkpoint_path,epoch=epoch,model=model,optimizer=optimizer)
        pbar.close()
        break


    print("Finished Training")

def load_generator(checkpoint_path=None,model=None,optimizer=None):
    # get epoch
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        model.apply(initialize_weights)
    return model,optimizer,start_epoch

def save_generator(checkpoint_path=None,epoch=0,model=None,optimizer=None):
    # Modify checkpoint saving directory based on the provided checkpoint_path
        if checkpoint_path:
            save_dir = os.path.dirname(checkpoint_path)
        else:
            save_dir = "models/generator"

        # Create save_dir if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        checkpoint_save_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_save_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ImageMerger model.')
    parser.add_argument('--checkpoint', type=str, help='Path to load the checkpoint from.')
    parser.add_argument('--freeze', action='store_true', help='Freeze feature extractor.')
    args = parser.parse_args()

    train(checkpoint_path=args.checkpoint)

from torch.utils.data import DataLoader
from dataset import EmojiDataset
import torch
import torch.nn as nn
import torch.optim as optim
from model import ImageMerger
from tqdm import tqdm


def main():
    device = torch.device("mps")

    model = ImageMerger().to(device)  # 'device' can be 'cuda', 'mps' or 'cpu'
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = EmojiDataset("data/emojiData.json")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Define hyperparameters
    num_epochs = 25
    print_every = 100  # Print training stats every 100 batches

    # Init progess bar
    pbar = tqdm(total=len(dataloader))

    for epoch in range(num_epochs):
        pbar.clear()
        running_loss = 0.0
        for i, (left_images, right_images, target_images) in enumerate(dataloader, 1):
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
           

        # Save model checkpoint after each epoch
        checkpoint_path = f"models/model_epoch_{epoch + 1}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        break

    pbar.close()

    print("Finished Training")


if __name__ == "__main__":
    main()

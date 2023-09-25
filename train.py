from torch.utils.data import DataLoader
from dataset import EmojiDataset
import torch
import torch.nn as nn
import torch.optim as optim
from generator_model import ImageMerger
from discriminator_model import Discriminator
from tqdm import tqdm
import argparse
from generator_train import load_generator,save_generator
from discriminator_train import load_discriminator,save_discriminator

def initialize_weights(layer):
    if isinstance(layer, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


def train(checkpoint_path_gen=None,checkpoint_path_dis=None,frezze_feature_extractor=False):
    # 'device' can be 'cuda', 'mps' or 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    generator = ImageMerger().to(device)
    discriminator = Discriminator().to(device)

    # Freeze feature extractor
    if frezze_feature_extractor:
        generator.set_feature_extractor_grad(frezze_feature_extractor)
        discriminator.set_feature_extractor_grad(frezze_feature_extractor)

    criterion = nn.BCELoss()
    optimizer_gen = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_dis = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # load generator
    generator,optimizer_gen,start_epoch_gen = load_generator(checkpoint_path=checkpoint_path_gen,model=generator,optimizer=optimizer_gen)

    # load discriminator
    discriminator,optimizer_dis,start_epoch_dis = load_discriminator(checkpoint_path=checkpoint_path_dis,model=discriminator,optimizer=optimizer_dis)

    # determine start epoch
    start_epoch = max(start_epoch_gen,start_epoch_dis)

    dataset = EmojiDataset("data/emojiData.json")
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Define hyperparameters
    num_epochs = 10
    print_every = 100  # Print training stats every 100 batches

    for epoch in range(start_epoch, num_epochs):
        # Init progess bar
        pbar = tqdm(total=len(dataloader))
        running_loss_gen = 0.0
        running_loss_dis = 0.0
        for i,(left_img, right_img, real_merged_img) in enumerate(dataloader):  # Assume dataloader is created from your EmojiDataset
            # Move data to device
            left_img, right_img, real_merged_img = (
                left_img.to(device),
                right_img.to(device),
                real_merged_img.to(device),
            )

            # Train discriminator
            optimizer_dis.zero_grad()
            
            fake_merged_img = generator(left_img, right_img)
            
            real_labels = torch.ones((batch_size, 1)).to(device)
            fake_labels = torch.zeros((batch_size, 1)).to(device)
            
            logits_real = discriminator(left_img, right_img, real_merged_img)
            logits_fake = discriminator(left_img, right_img, fake_merged_img.detach())
            
            loss_real = criterion(logits_real, real_labels)
            loss_fake = criterion(logits_fake, fake_labels)
            
            loss_dis = (loss_real + loss_fake) / 2
            loss_dis.backward()
            optimizer_dis.step()
            
            # Train generator
            optimizer_gen.zero_grad()
            
            logits_fake = discriminator(left_img, right_img, fake_merged_img)
            loss_gen = criterion(logits_fake, real_labels)  # We want the generator to fool the discriminator
            
            loss_gen.backward()
            optimizer_gen.step()


            # Print statistics
            running_loss_dis += loss_dis.item()
            running_loss_gen += loss_gen.item()
            if i % print_every == 0:
                running_loss_dis = 0.0
                running_loss_gen = 0.0
            else:
                pbar.set_postfix(loss_gem=running_loss_gen / (i%print_every),loss_dis=running_loss_dis / (i%print_every))
            pbar.update(1)
            pbar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")

        # Save model checkpoint
        save_generator(checkpoint_path=checkpoint_path_gen,epoch=epoch,model=generator,optimizer=optimizer_gen)
        save_discriminator(checkpoint_path=checkpoint_path_dis,epoch=epoch,model=discriminator,optimizer=optimizer_dis)

        pbar.close()
        break

    print("Finished Training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ImageMerger model.')
    parser.add_argument('--checkpoint-gen', type=str, help='Path to load the checkpoint from.')
    parser.add_argument('--checkpoint-dis', type=str, help='Path to load the checkpoint from.')
    parser.add_argument('--freeze', action='store_true', help='Freeze feature extractor.')
    args = parser.parse_args()

    train(checkpoint_path_gen=args.checkpoint_gen,checkpoint_path_dis=args.checkpoint_dis,frezze_feature_extractor=args.freeze)

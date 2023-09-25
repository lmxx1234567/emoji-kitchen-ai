from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from google_emoji import getEmojiUrl, googleRequestUrl
from PIL import Image
import requests
import os
import json
from google_emoji import EmojiCombo
from tqdm import tqdm
import torch
import random


class EmojiDataset(Dataset):
    def __init__(self, data_path, size=128, cache_dir="./data"):
        self.session = requests.Session()
        self.size = size
        with open(data_path, "r") as f:
            data = json.load(f)
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),  # Resizing to a common size
                transforms.ToTensor(),
            ]
        )
        self.combos = [
            EmojiCombo(**item) for _, values in data.items() for item in values
        ]
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)  # Create cache directory if it doesn't exist

    def __len__(self):
        return len(self.combos)

    def __getitem__(self, index):
        combo = self.combos[index]
        left_img_url = getEmojiUrl(combo.leftEmoji, self.size)
        right_img_url = getEmojiUrl(combo.rightEmoji, self.size)
        merged_img_url = googleRequestUrl(combo)
        left_img = self.transform(
            self.get_or_cache_image(left_img_url, combo.leftEmoji, "emoji")
        )
        right_img = self.transform(
            self.get_or_cache_image(right_img_url, combo.rightEmoji, "emoji")
        )
        merged_img = self.transform(
            self.get_or_cache_image(
                merged_img_url, f"{combo.leftEmoji}_{combo.rightEmoji}", "merged"
            )
        )
        return left_img, right_img, merged_img

    def get_cached_image_path(self, emoji_id, prefix=""):
        return os.path.join(self.cache_dir, prefix, f"{emoji_id}.png")

    def get_or_cache_image(self, url, emoji_id, prefix=""):
        cached_path = self.get_cached_image_path(emoji_id, prefix)
        if os.path.exists(cached_path):
            # Load from cache
            try:
                img = Image.open(cached_path)
                return img.convert('RGB')
            except:
                return self.get_image(url, cached_path)
        else:
            return self.get_image(url, cached_path)


    def get_image(self, url, cached_path):
        # Fetch image, cache it, and return
        with self.session.get(url, stream=True) as response:
            response.raise_for_status()  # Raise error for failed requests
            with Image.open(response.raw) as img:
                subdir = os.path.dirname(cached_path)
                if not os.path.exists(subdir):
                    os.mkdir(subdir)
                img.save(cached_path)
                return img.convert('RGB')

class EmojiWithFakeDataset(Dataset):
    def __init__(self, data_path, size=128, cache_dir="./data",fake_rate=0.5):
        self.session = requests.Session()
        self.size = size
        with open(data_path, "r") as f:
            data = json.load(f)
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),  # Resizing to a common size
                transforms.ToTensor(),
            ]
        )
        self.combos = [
            EmojiCombo(**item) for _, values in data.items() for item in values
        ]
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)  # Create cache directory if it doesn't exist

        self.fake_rate = fake_rate
        self.len = int(len(self.combos)/(1-self.fake_rate))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # Check if we should return a fake image
        if index < len(self.combos) :
            combo = self.combos[index]
            left_img_url = getEmojiUrl(combo.leftEmoji, self.size)
            right_img_url = getEmojiUrl(combo.rightEmoji, self.size)
            merged_img_url = googleRequestUrl(combo)
            combo = self.combos[index]
            left_img = self.transform(
                self.get_or_cache_image(left_img_url, combo.leftEmoji, "emoji")
            )
            right_img = self.transform(
                self.get_or_cache_image(right_img_url, combo.rightEmoji, "emoji")
            )
            merged_img = self.transform(
                self.get_or_cache_image(
                    merged_img_url, f"{combo.leftEmoji}_{combo.rightEmoji}", "merged"
                )
            )
            return left_img, right_img, merged_img, torch.tensor([1.0])
        else:
            left_img, right_img, merged_img = self.get_fake_images()
            return left_img, right_img, merged_img, torch.tensor([0.0])
        
    def get_fake_images(self):
        # Get three random indexes
        indexes = random.sample(range(len(self.combos)), 3)

        # Get the three images
        left_emoji = self.combos[indexes[0]].leftEmoji
        right_emoji = self.combos[indexes[1]].rightEmoji
        merged_combo = self.combos[indexes[2]]
        left_img_url = getEmojiUrl(left_emoji, self.size)
        right_img_url = getEmojiUrl(right_emoji, self.size)
        merged_img_url = googleRequestUrl(merged_combo)
        left_img = self.transform(
            self.get_or_cache_image(left_img_url, left_emoji, "emoji")
        )
        right_img = self.transform(
            self.get_or_cache_image(right_img_url,right_emoji, "emoji")
        )
        merged_img = self.transform(
            self.get_or_cache_image(
                merged_img_url, f"{merged_combo.leftEmoji}_{merged_combo.rightEmoji}", "merged"
            )
        )
        return left_img, right_img, merged_img

    def get_cached_image_path(self, emoji_id, prefix=""):
        return os.path.join(self.cache_dir, prefix, f"{emoji_id}.png")

    def get_or_cache_image(self, url, emoji_id, prefix=""):
        cached_path = self.get_cached_image_path(emoji_id, prefix)
        if os.path.exists(cached_path):
            # Load from cache
            try:
                img = Image.open(cached_path)
                return img.convert('RGB')
            except:
                return self.get_image(url, cached_path)
        else:
            return self.get_image(url, cached_path)


    def get_image(self, url, cached_path):
        # Fetch image, cache it, and return
        with self.session.get(url, stream=True) as response:
            response.raise_for_status()  # Raise error for failed requests
            with Image.open(response.raw) as img:
                subdir = os.path.dirname(cached_path)
                if not os.path.exists(subdir):
                    os.mkdir(subdir)
                img.save(cached_path)
                return img.convert('RGB')



if __name__ == "__main__":
    dataset = EmojiDataset('data/emojiData.json')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for index in tqdm(dataloader):
        pass


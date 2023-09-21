import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from google_emoji import getEmojiUrl, googleRequestUrl
from PIL import Image
import requests
import os
from google_emoji import EmojiCombo


class EmojiDataset(Dataset):
    def __init__(self, data_path, size=128, cache_dir="./data"):
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
            img = Image.open(cached_path)
            return img.convert('RGB')
        else:
            # Fetch image, cache it, and return
            with requests.get(url, stream=True) as response:
                response.raise_for_status()  # Raise error for failed requests
                with Image.open(response.raw) as img:
                    subdir = os.path.dirname(cached_path)
                    if not os.path.exists(subdir):
                        os.mkdir(subdir)
                    img.save(cached_path)
                    return img.convert('RGB')


# if __name__ == "__main__":
#     dataset = EmojiDataset('emoji-kitchen/data/emojiData.json')
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

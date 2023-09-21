rootUrl = "https://www.gstatic.com/android/keyboard/emojikitchen"


class EmojiCombo:
    def __init__(self, date: str, leftEmoji: str, rightEmoji: str):
        self.date = date
        self.leftEmoji = leftEmoji
        self.rightEmoji = rightEmoji


def googleRequestUrlEmojiPart(emoji: str) -> str:
    return "-".join([f"u{part.lower()}" for part in emoji.split("-")])


def googleRequestUrlEmojiFilename(combo: EmojiCombo) -> str:
    return f"{googleRequestUrlEmojiPart(combo.leftEmoji)}_{googleRequestUrlEmojiPart(combo.rightEmoji)}.png"


def googleRequestUrl(combo: EmojiCombo) -> str:
    return f"{rootUrl}/{combo.date}/{googleRequestUrlEmojiPart(combo.leftEmoji)}/{googleRequestUrlEmojiFilename(combo)}"


def getEmojiUrl(emoji: str, size=128) -> str:
    return f'https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/{size}/emoji_u{"_".join([part for part in emoji.split("-") if part!="fe0f"])}.png'


# if __name__ == "__main__":
#     combo = EmojiCombo("20210521", "1fa84", "2615")
#     print(googleRequestUrl(combo))
#     print(getEmojiUrl("1f636-200d-1f32b-fe0f"))

# Emoji Kitchen AI

An AI-driven solution to merge and generate new emoji combinations.

## Introduction

`emoji-kitchen-ai` utilizes deep learning, specifically Convolutional Neural Networks (CNNs), to intelligently merge two emojis into one, producing creative and novel combinations. Built with PyTorch, this project leverages pre-trained models like ResNet50 for feature extraction and a custom head for the merging task.

## Installation

### Prerequisites

- Python 3.x
- PyTorch and torchvision
- Any other dependencies (mention specific versions if necessary)

### Steps

1. Clone the repository:
   ```
   git clone https://github.com/lmxx1234567/emoji-kitchen-ai.git
   cd emoji-kitchen-ai
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation**
   - Ensure you have a dataset ready based on [xsalazar/emoji-kitchen](https://github.com/xsalazar/emoji-kitchen/blob/main/scripts/emojiOutput.json).
   - If using custom data, adjust the paths in the configuration or script.

2. **Training**
   - Run the training script:
     ```
     python train.py
     ```

3. **Prediction**
   - To predict and visualize new emoji combinations:
     ```
     python predict.py --image1 path_to_first_image --image2 path_to_second_image
     ```

## Results

Showcase some of the best results from the model here. You can provide images or GIFs of the merged emojis.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/lmxx1234567/emoji-kitchen-ai/issues). 

## License

This project is [MIT](LICENSE) licensed.

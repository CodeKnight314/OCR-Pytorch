import json
import argparse
from model import OCRModel
from glob import glob
from PIL import Image
import torch
import os
import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

def decode_logits_output(prediction: torch.Tensor, idx2char: dict):
    """
    Args:
        prediction (torch.Tensor): Predicted logits tensor with shape [batch_size, seq_len, vocab_size]
    """
    prediction = prediction.argmax(dim=-1)
    batch_string = []
    for sequence in prediction:
        text = ''.join(idx2char[idx] for idx in np.array(sequence.cpu()))
        text = ''.join(char for char in text if char != idx2char[0])
        batch_string.append(text)
    return batch_string

def inference(model: OCRModel, char_dict: dict, idx_dict: dict, img_dir: str, output: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    img_directory = glob(os.path.join(img_dir, "*.png"))
    transforms = T.Compose([T.Resize((64, 128)), T.ToTensor()])

    for img_path in tqdm(img_directory):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms(img).to(device)

        with torch.no_grad():
            prediction = model(img_tensor.unsqueeze()).squeeze()

        text = decode_logits_output(prediction.unsqueeze(0), idx_dict)[0]

        plt.imshow(img)
        plt.title(text)
        plt.axis('off')
        
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output, img_name + ".png")
        plt.savefig(output_path)
        plt.close()
    
    print("All predictions have been saved to specified output directory")
        
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to image directory containing images")
    parser.add_argument("--weight", type=str, required=True, help="Path to model weights")
    parser.add_argument("--output", type=str, required=True, help="Path to saving inference results")
    parser.add_argument("--json", type=str, required=True, help="Path to char-idx.json")
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        loaded_dicts = json.load(f)
    char_dict = loaded_dicts['char_dict']
    idx_dict = loaded_dicts['idx_dict']
    
    vocab_size = len(char_dict)
    model = OCRModel(vocab_size=vocab_size)
    model.load_state_dict(torch.load(args.weight, weights_only=True))
    
    inference(model, char_dict, idx_dict, args.dir, args.output)
    
    
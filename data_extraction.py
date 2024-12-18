# data_extraction.py
import os
import sys
import torch
from PIL import Image
import glob
import pickle
import argparse
from torchvision import transforms
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
import clip
import logging

def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def crop_to_square(img):
    size = 512
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
        ]
    )
    return image_transforms(img)

class CLIPModel(object):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = clip.tokenize

    def text_emb(self, text_ls):
        if isinstance(text_ls, str):
            text_ls = [text_ls]
        text = self.tokenizer(text_ls, truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features

    def img_emb(self, img):
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features

    def __call__(self, image, text, softmax=False):
        if isinstance(text, str):
            text = [text]

        if isinstance(image, list):
            image = [self.preprocess(i).unsqueeze(0).to(self.device) for i in image]
            image = torch.cat(image)
        else:
            image = self.preprocess(image).unsqueeze(0).to(self.device)

        text = self.tokenizer(text).to(self.device)

        if softmax:
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            return probs
        else:
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
                s = similarity[0][0]

            return s

def main():
    # Setup logging
    log_path = os.path.join(args.outdir, 'data_extraction.log')
    setup_logging(log_path)
    logging.info(f"Starting data extraction for concept: {args.concept}")
    
    clip_model = CLIPModel()
    data_dir = args.directory

    source_concept = args.concept
    os.makedirs(args.outdir, exist_ok=True)
    all_data = glob.glob(os.path.join(data_dir, "*.p"))
    res_ls = []
    for idx, cur_data_f in enumerate(all_data):
        try:
            cur_data = pickle.load(open(cur_data_f, "rb"))
            cur_img = Image.fromarray(cur_data["img"])
            cur_text = cur_data["text"]

            cur_img = crop_to_square(cur_img)
            score = clip_model(cur_img, f"a photo of a {source_concept}")
            if score > 0.24:
                res_ls.append((cur_img, cur_text))
        except Exception as e:
            logging.error(f"Error processing file {cur_data_f}: {e}")

    logging.info(f"Total candidates found: {len(res_ls)}")

    if len(res_ls) < args.num:
        logging.error("Not enough data from the source concept to select from. Please add more in the folder.")
        raise Exception("Not enough data from the source concept to select from. Please add more in the folder.")

    all_prompts = [d[1] for d in res_ls]
    text_emb = clip_model.text_emb(all_prompts)
    text_emb_target = clip_model.text_emb(f"a photo of a {source_concept}")
    text_emb_np = text_emb.cpu().float().numpy()
    text_emb_target_np = text_emb_target.cpu().float().numpy()
    res = cosine_similarity(text_emb_np, text_emb_target_np).reshape(-1)
    candidate = np.argsort(res)[::-1][:300]
    random_selected_candidate = random.sample(list(candidate), args.num)
    final_list = [res_ls[i] for i in random_selected_candidate]

    for i, data in enumerate(final_list):
        img, text = data
        cur_data = {
            "img": np.array(img),
            "text": text,
        }
        with open(os.path.join(args.outdir, f"{i}.p"), "wb") as pf:
            pickle.dump(cur_data, pf)
        if (i + 1) % 50 == 0:
            logging.info(f"Selected {i + 1} images for poisoning")
    
    logging.info("Data extraction and selection completed successfully")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, required=True,
                        help="Directory containing .p files")
    parser.add_argument('-od', '--outdir', type=str, required=True,
                        help="Output directory for selected .p files")
    parser.add_argument('-n', '--num', type=int, default=100,
                        help="Number of images to select")
    parser.add_argument('-c', '--concept', type=str, required=True,
                        help="Source concept name")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()

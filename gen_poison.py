# gen_poison.py
import os
import glob
import pickle
import argparse
from opt import PoisonGeneration
from PIL import Image
import numpy as np
import logging

def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def main():
    # Setup logging
    log_path = os.path.join(args.outdir, 'poisoning.log')
    setup_logging(log_path)
    
    logging.info(f"Starting poisoning with target concept: {args.target_name}")
    
    poison_generator = PoisonGeneration(
        target_concept=args.target_name,
        device="cuda",
        eps=args.eps,
        num_opt_steps=args.num_opt_steps,
        sd_steps=args.sd_steps
    )
    
    all_data_paths = glob.glob(os.path.join(args.directory, "*.p"))
    all_imgs = []
    all_texts = []
    
    for f in all_data_paths:
        data = pickle.load(open(f, "rb"))
        all_imgs.append(Image.fromarray(data['img']))
        all_texts.append(data['text'])
    
    poisoned_images = poison_generator.generate_all(all_imgs)
    os.makedirs(args.outdir, exist_ok=True)
    
    for idx, cur_img in enumerate(poisoned_images):
        cur_data = {"text": all_texts[idx], "img": np.array(cur_img)}
        with open(os.path.join(args.outdir, f"{idx}.p"), "wb") as pf:
            pickle.dump(cur_data, pf)
        if (idx + 1) % 50 == 0:
            logging.info(f"Poisoned {idx + 1} images")
    
    logging.info("Finished poisoning all images")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, required=True,
                        help="Directory containing selected .p files")
    parser.add_argument('-od', '--outdir', type=str, required=True,
                        help="Output directory for poisoned .p files")
    parser.add_argument('-e', '--eps', type=float, default=0.05,
                        help="Maximum perturbation strength")
    parser.add_argument('-t', '--target_name', type=str, required=True,
                        help="Target concept name")
    parser.add_argument('-n', '--num_opt_steps', type=int, default=500,
                        help="Number of optimization steps")
    parser.add_argument('--sd_steps', type=int, default=50,
                        help="Number of Stable Diffusion inference steps")
    args = parser.parse_args()
    main()

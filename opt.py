import os
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import torch.utils.data
from einops import rearrange
from PIL import Image
from torchvision import transforms


class PoisonGeneration(object):
    def __init__(self, target_concept, device, eps=0.05, num_opt_steps=500, sd_steps=50):
        self.eps = eps
        self.target_concept = target_concept
        self.device = device
        self.num_opt_steps = num_opt_steps
        self.sd_steps = sd_steps
        self.full_sd_model = self.load_model()
        self.transform = self.resizer()

        # Pre-generate target latent once
        self.target_latent = self.generate_target_latent("A photo of a {}".format(self.target_concept))

    def resizer(self):
        image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
            ]
        )
        return image_transforms

    def load_model(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            safety_checker=None,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipeline = pipeline.to(self.device)
        return pipeline

    def generate_target_image(self, prompt):
        torch.manual_seed(123)  # ensuring consistency
        with torch.no_grad():
            # Use autocast for SD inference
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                target_imgs = self.full_sd_model(
                    prompt,
                    guidance_scale=7.5,
                    num_inference_steps=self.sd_steps,
                    height=512,
                    width=512
                ).images
        return target_imgs[0]

    def generate_target_latent(self, prompt):
        target_image = self.generate_target_image(prompt)
        target_tensor = img2tensor(target_image).to(self.device).half()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                target_latent = self.get_latent(target_tensor)
        return target_latent

    def get_latent(self, tensor):
        # The VAE encode step can also benefit from autocast
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            latent_features = self.full_sd_model.vae.encode(tensor).latent_dist.mean
        return latent_features

    def generate_one(self, pil_image):
        resized_pil_image = self.transform(pil_image)
        source_tensor = img2tensor(resized_pil_image).to(self.device).half()

        target_latent = self.target_latent
        modifier = torch.zeros_like(source_tensor, device=self.device, dtype=source_tensor.dtype)

        t_size = self.num_opt_steps
        max_change = self.eps / 0.5
        step_size = max_change

        # Gradient-based optimization
        for i in range(t_size):
            modifier.requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
                adv_latent = self.get_latent(adv_tensor)

                loss = (adv_latent - target_latent).norm()
                tot_loss = loss.sum()

            grad = torch.autograd.grad(tot_loss, modifier)[0]

            # Update modifier outside autocast to ensure correct casting
            modifier = modifier - torch.sign(grad) * step_size
            modifier = torch.clamp(modifier, -max_change, max_change)
            modifier = modifier.detach()

            if i % (t_size // 5) == 0:
                print(f"# Iter: {i}\tLoss: {loss.mean().item():.3f}")

        final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)
        final_img = tensor2img(final_adv_batch)
        return final_img

    def generate_all(self, image_list, target_concept):
        res_imgs = []
        for idx, image_f in enumerate(image_list):
            cur_img = image_f.convert('RGB')
            perturbed_img = self.generate_one(cur_img)
            res_imgs.append(perturbed_img)
        return res_imgs

def img2tensor(cur_img):
    cur_img = cur_img.resize((512, 512), resample=Image.Resampling.BICUBIC)
    cur_img = np.array(cur_img)
    img = (cur_img / 127.5 - 1.0).astype(np.float32)
    img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img).unsqueeze(0)
    return img

def tensor2img(cur_img):
    cur_img = torch.clamp((cur_img.detach() + 1.0) / 2.0, min=0.0, max=1.0)
    cur_img = 255. * rearrange(cur_img[0], 'c h w -> h w c').cpu().numpy()
    cur_img = Image.fromarray(cur_img.astype(np.uint8))
    return cur_img

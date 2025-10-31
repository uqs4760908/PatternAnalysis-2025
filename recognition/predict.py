"""
A simple script to generate some images.
"""

from pathlib import Path

from PIL import Image
from dataset import NC_LABEL, AD_LABEL, NUM_CLASS
from train import VAEController, DiffusionModelController, VAE_CONFIG, DIFFUSION_CONFIG
from modules import DiffusionModel, DiffusionSampler, VAE
from torch.nn import functional as F
from config import ImageInfo
import numpy as np
import argparse
import torch
import tqdm


BATCH_SIZE = 8


parser = argparse.ArgumentParser(description="Generate brain images")
parser.add_argument("--num-nc-images", default=16, type=int,
                    help="How many images of normal cofnitive brains to generate")
parser.add_argument("--num-ad-images", default=16, type=int,
                    help="How many images of alzheimer disease brains to generate")
parser.add_argument("--vae-weights", 
                    default=VAEController.MODEL_PATH, type=str,
                    help="Path to pretrained VAE weights")
parser.add_argument("--diffusion-weights", 
                    default=DiffusionModelController.EMA_MODEL_PATH,
                    type=str,
                    help="Path to pretrained diffusion model weights")
parser.add_argument("--output-path", "-o", default="images", type=str,
                    help="Path to save generated images")
parser.add_argument("--save-per-steps", default=DIFFUSION_CONFIG.denoise_steps, type=int,
                    help="Save intermediate images per N steps")
args = parser.parse_args()


device = torch.accelerator.current_accelerator() or torch.get_default_device()


# Load pretrained model
# Models are usually trained on CUDA. In case CUDA is not available during inference,
# load weights to CPU then copy to whatever device available
vae_weights = torch.load(args.vae_weights, weights_only=True, map_location="cpu")
# hard coded info for ANDI, since we might not have the dataset during inference
image_info = ImageInfo(size=(240, 256), depth=1)
vae = VAE(VAE_CONFIG, image_info)
vae.load_state_dict(vae_weights)
vae = vae.to(device)

diffusion_model = DiffusionModel(DIFFUSION_CONFIG, VAE_CONFIG.latent_dim, NUM_CLASS)
diffusion_weights = torch.load(args.diffusion_weights, weights_only=True, map_location="cpu")
diffusion_model.load_state_dict(diffusion_weights)
diffusion_model = diffusion_model.to(device)

vae.eval()
diffusion_model.eval()


if not Path(args.output_path).exists():
    Path(args.output_path).mkdir()


sampler = DiffusionSampler(DIFFUSION_CONFIG).to(device)


def gen(total: int, label: torch.Tensor, tag: str):
    size = VAE_CONFIG.output_size(image_info.size)

    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    for batch in range(n_batches):
        num_images = min(BATCH_SIZE, total - batch * BATCH_SIZE)

        generator = sampler.generate_with_steps(
                num_images, 
                size, 
                VAE_CONFIG.latent_dim, 
                label, 
                diffusion_model)
        print(f"Batch [{batch + 1}/{n_batches}]")
        for step, latents in enumerate(tqdm.tqdm(generator, total=DIFFUSION_CONFIG.denoise_steps)):
            if (step // args.save_per_steps) == (step + 1) // args.save_per_steps:
                continue

            images = vae.decode(latents).numpy(force=True)
            images = images * 255
            images = images.astype(np.uint8)
            for i, image in enumerate(images):
                image = image.squeeze()
                imageobj = Image.fromarray(image)
                imageobj.save(f"{args.output_path}/{tag}-{i + batch * BATCH_SIZE}-{step}.jpg")


# Sample
with torch.inference_mode(), torch.autocast(device_type=device.type):
    print("Generating AD images")
    label = F.one_hot(torch.tensor([AD_LABEL]), NUM_CLASS).float().to(device)
    gen(args.num_ad_images, label, tag="ad")

    print("Generating NC images")
    label = F.one_hot(torch.tensor([NC_LABEL]), NUM_CLASS).float().to(device)
    gen(args.num_nc_images, label, tag="nc")

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from dataset import ANDIDataset
from modules import VAE
from train import VAE_CONFIG
from matplotlib import pyplot as plt
import torch
import umap


NUM_IMAGES = 8192
BATCH_SIZE = 8

device = torch.accelerator.current_accelerator() or torch.get_default_device()


dataset = ANDIDataset(Path("./data"))
vae = VAE(VAE_CONFIG, dataset.image_info)

vae_weights = torch.load("./vae.pt", weights_only=True, map_location="cpu")
vae.load_state_dict(vae_weights)
vae = vae.to(device)
vae.eval()

images: list[torch.Tensor] = []
labels: list[int] = []
for i in range(NUM_IMAGES):
    image, label = dataset.train_dataset[i]
    images.append(image)
    labels.append(label)

for i in range(NUM_IMAGES):
    image, label = dataset.train_dataset[i + len(dataset.train_dataset.ad_images)]
    images.append(image)
    labels.append(label)


with torch.inference_mode(), torch.autocast(device_type=device.type):
    latents = []

    for i in range(NUM_IMAGES * 2 // BATCH_SIZE):
        chunk = images[i:i + BATCH_SIZE]
        image_tensor = torch.stack(chunk).to(device)
        latent = vae.encode(image_tensor).flatten(1, 3).numpy(force=True)
        latents.extend(latent)

scaled_latents = StandardScaler().fit_transform(latents, labels)
reducer = umap.UMAP()
embedding = reducer.fit_transform(scaled_latents)

colors = [(1, 0, 0), (0, 1, 0)]
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[colors[i] for i in labels],
    s=5) # RGB values for each class
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of latent images', fontsize=24);
plt.savefig("unet.jpg")

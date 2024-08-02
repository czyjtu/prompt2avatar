# write pytorch lightning callbacks for saving best model and logging to wandb 

import numpy as np
import numpy.typing as npt 
import torch 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from sklearn.decomposition import PCA
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from utils.dataset import ProcessedDataset
from tqdm import tqdm 


def pca_combined_features_v1(dataset: ProcessedDataset, pcas: dict[str, PCA] | None = None):
    component_feature_names = [
        "layer1-2-bn3",
        "layer2-12-bn3",
        "layer3-29-bn3",
        "layer4-2-bn3",
        "arcface-r100",
        "landmarks"
    ]

    if pcas is None:
        pcas = {
            name: PCA(n_components=0.95).fit(dataset.model2normalized_embeddings[name]) 
            for name in component_feature_names
        }
    
    features = []
    for name in sorted(component_feature_names):
        features.append(pcas[name].transform(dataset.model2normalized_embeddings[name]))
    stacked_features = np.hstack(features)
    return stacked_features, pcas 

def batch_predict(model, input, batch_size):
    partial_output = []
    total_batches = len(input) // batch_size
    for batch_start in tqdm(range(0, len(input), batch_size), total=total_batches):
        batch = input[batch_start: batch_start + batch_size]
        with torch.no_grad():
            partial_output.append(model(batch))
    return partial_output

def get_clip_embeddings(images: np.ndarray, device: str = "cpu", dtype=torch.float32, batch_size: int=32):
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(
            device, dtype=dtype
        )
    clip_image_processor = CLIPImageProcessor()
    def predict(batch):
        clip_images = clip_image_processor(images=batch, return_tensors="pt").pixel_values
        clip_images = clip_images.to(device, dtype=dtype)
        return image_encoder(clip_images).image_embeds
    
    partials = batch_predict(predict, images, batch_size)
    return torch.concatenate(partials)


from torchvision import transforms as VIT


class AugmentationDataset(Dataset):
    def __init__(self, images, transform=None):
        self.data = images
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image
    
if __name__ == "__main__":
    images = np.zeros((100, 112, 112, 3))
    emb = get_clip_embeddings(images)
    print(emb.shape)
# write pytorch lightning callbacks for saving best model and logging to wandb 

import numpy as np
import numpy.typing as npt 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.decomposition import PCA

from utils.dataset import ProcessedDataset


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
        features.append(pcas[name].transform(dataset.model2embeddings[name]))
    stacked_features = np.hstack(features)
    return stacked_features, pcas 

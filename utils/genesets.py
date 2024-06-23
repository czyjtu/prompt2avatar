from pathlib import Path
import sys 


sys.path.append(Path(__file__).resolve().parent.parent.__str__())
print(sys.path)

import numpy as np

from utils.dataset import ProcessedDataset 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_GENESET_DIR = DATA_DIR / "test_geneset_embeddings"
TEST_GENESET_GENES = ["gene_forehead_brow_height.forehead_brow_height_pos", "gene_jaw_height.jaw_height_pos", "skin_color[1]"]

def load_geneset_dataset(geneset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    geneset_path, genes_to_predict = {
        "test": (TEST_GENESET_DIR, TEST_GENESET_GENES)
    }[geneset_name]
    dataset = ProcessedDataset(geneset_path)
    genes_vectorized = [dna.asarray(list(genes_to_predict)) for dna in dataset.dna]
    genes_vectorized = np.stack(genes_vectorized)
    X = genes_vectorized
    Y = dataset.model2normalized_embeddings["arcface-r100"]

    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)

    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    train_images = np.array([dataset.alligned_images[i] for i in train_indices])
    test_images = np.array([dataset.alligned_images[i] for i in test_indices])
    scaler_x = MinMaxScaler()
    X_train_sc = scaler_x.fit_transform(X_train)
    X_test_sc = scaler_x.transform(X_test)

    return X_train_sc, X_test_sc, Y_train, Y_test, train_images, test_images, scaler_x

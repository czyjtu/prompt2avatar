from pathlib import Path 
from tqdm import tqdm 
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.preprocessing import normalize
import pandas as pd 
from utils.dna import Genes

_model_name = str


@dataclass 
class ProcessedDataset:
    path: Path
    alligned_images: np.ndarray = field(init=False)
    paths: np.ndarray = field(init=False)
    labels: np.ndarray = field(init=False)
    model2embeddings: dict[_model_name, np.ndarray] = field(init=False)
    model2normalized_embeddings: dict[_model_name, np.ndarray] = field(init=False)
    dna: list[Genes] = field(init=False)

    def __post_init__(self):
        self.alligned_images = np.load(self.path / "alligned_images.npy")
        self.paths = np.load(self.path / "alligned_paths.npy", allow_pickle=True)
        self.labels = np.load(self.path / "alligned_labels.npy", allow_pickle=True)
        self.model2embeddings = self._get_embeddings()
        self.model2normalized_embeddings = {model_name: normalize(embeddings) for model_name, embeddings in self.model2embeddings.items()}
        self.dna = [Genes.from_ck_string(dna) for dna in np.load(self.path / "dnas.npy", allow_pickle=True)]

    def _get_embeddings(self):
        model2embeddings = {}
        embeddings_files = list(self.path.glob('embeddings*.npy'))
        for embeddings_file in tqdm(embeddings_files):
            model_name = "-".join(embeddings_file.stem.split('_')[1:])
            model2embeddings[model_name] = np.load(embeddings_file)
        return model2embeddings
    
    def asframe(self, is_fake: bool | None) -> dict[_model_name, pd.DataFrame]:
        labels = list(map(str, self.labels))
        paths = list(map(str, self.paths))
        model2df = {}
        for model in self.model2embeddings:
            init_dict = {
                "embeddings": list(self.model2embeddings[model]),
                "embeddings_l2norm": list(self.model2normalized_embeddings[model]),
                "labels": labels,
                "fnames": paths,
            }
            if is_fake is not None:
                init_dict["is_fake"] = [int(is_fake)] * len(self.model2embeddings[model])

            df = pd.DataFrame(init_dict)
            model2df[model] = df
        return model2df

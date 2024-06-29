from pathlib import Path 
from tqdm import tqdm 
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.preprocessing import normalize
import pandas as pd 
from utils.dna import Genes, create_template_from_genes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

_model_name = str
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_GENESET_DIR = DATA_DIR / "test_geneset_embeddings"
TEST_GENESET_GENES = ["gene_forehead_brow_height.forehead_brow_height_pos", "gene_jaw_height.jaw_height_pos", "skin_color[1]"]
EASY_GENESET_DIR = DATA_DIR / "easy_geneset_embeddings"
EASY_GENESET_GENES = [
    'gene_forehead_brow_height.forehead_brow_height_pos',
    'gene_bs_cheek_forward.cheek_forward_pos',
    'gene_chin_height.chin_height_pos',
    'gene_head_height.head_height_pos',
    'gene_jaw_height.jaw_height_pos',
    'gene_mouth_upper_lip_size.mouth_upper_lip_size_pos',
    'gene_mouth_height.mouth_height_pos',
    'gene_mouth_width.mouth_width_pos',
    'gene_bs_nose_tip_angle.nose_tip_angle_pos',
    'gene_bs_nose_height.nose_height_pos',
    'gene_eye_angle.eye_angle_pos',
    'gene_eye_distance.eye_distance_pos',
    'skin_color[1]',
]
_GENESETS = {
    "test": (TEST_GENESET_DIR, TEST_GENESET_GENES),
    "easy": (EASY_GENESET_DIR, EASY_GENESET_GENES)
}


@dataclass 
class ProcessedDataset:
    path: Path
    alligned_images: np.ndarray = field(init=False)
    paths: np.ndarray = field(init=False)
    labels: np.ndarray = field(init=False)
    model2embeddings: dict[_model_name, np.ndarray] = field(init=False)
    model2normalized_embeddings: dict[_model_name, np.ndarray] = field(init=False)
    dna: list[Genes] | None = field(init=False, default=None)

    def __post_init__(self):
        self.alligned_images = np.load(self.path / "alligned_images.npy")
        self.paths = np.load(self.path / "alligned_paths.npy", allow_pickle=True)
        self.labels = np.load(self.path / "alligned_labels.npy", allow_pickle=True)
        self.model2embeddings = self._get_embeddings()
        self.model2normalized_embeddings = {model_name: normalize(embeddings) for model_name, embeddings in self.model2embeddings.items()}
        if (self.path / "dnas.npy").exists():
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
    

def load_geneset_dataset(geneset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    geneset_path, genes_to_predict = _GENESETS[geneset_name]
    dataset = ProcessedDataset(geneset_path)
    genes_vectorized = [dna.asarray(list(genes_to_predict)) for dna in dataset.dna]
    genes_vectorized = np.stack(genes_vectorized)
    X = genes_vectorized
    Y = dataset.model2normalized_embeddings["arcface-r100"]
    (X_train, Y_train, train_images), (X_test, Y_test, test_images) = split_by_indices(
        [X, Y, np.array(dataset.alligned_images)], 0.1
    )
    (X_train, Y_train, train_images), (X_val, Y_val, val_images) = split_by_indices([X_train, Y_train, train_images], 0.1)
    scaler_x = MinMaxScaler()
    X_train_sc = scaler_x.fit_transform(X_train)
    X_test_sc = scaler_x.transform(X_test)
    X_val_sc = scaler_x.transform(X_val)
    return X_train_sc, X_val_sc, X_test_sc, Y_train, Y_val, Y_test, train_images, val_images, test_images, scaler_x

def split_by_indices(arrs: list[np.ndarray], test_size: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    indices = np.arange(len(arrs[0]))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    arrs_train = [array[train_indices] for array in arrs]
    arrs_test = [array[test_indices] for array in arrs]
    return arrs_train, arrs_test

def save_predicitons(preds: np.ndarray, scaler: MinMaxScaler, path: Path, geneset_name: str) -> None:
    def ypred_to_dnas(Y_pred, predicted_genes, dataset_dna_template):
        genes = [Genes.from_array(Y_pred[i], predicted_genes, dataset_dna_template) for i in range(len(Y_pred))]
        return genes

    def save_dnas(genes: list[Genes], path: Path):
        path.mkdir(parents=True, exist_ok=True)
        for i, gene in enumerate(genes):
            with open(path / f'gene_{i}.txt', 'w') as f:
                f.write(gene.to_ck_string())

    geneset_path, genes_to_predict = _GENESETS[geneset_name]
    dataset = ProcessedDataset(geneset_path)
    template = create_template_from_genes(dataset.dna)
    save_dnas(ypred_to_dnas(scaler.inverse_transform(preds), genes_to_predict, template), path)

def get_template_from_geneset(geneset_name: str) -> dict:
    geneset_path, genes_to_predict = _GENESETS[geneset_name]
    dataset = ProcessedDataset(geneset_path)
    template = create_template_from_genes(dataset.dna)
    return template


def freeze_genes(base_dataset: list[Genes], unlocked_genes: list[str]):
    # what does this function do?
    """This function creates new dataset with the same genes as the base dataset, but the genes that are not in the given list are replaced by the mean value."""
    template = create_template_from_genes(base_dataset)
    for gene in unlocked_genes:
        for k in template.keys():
            if gene in k:
                del template[k]
                print("Deleted: ", k)
                break 
    new_dataset = []
    hashes = set()
    for gene in base_dataset:
        gene_dict = gene.flatten()
        gene_dict.update(template)
        new_dna = Genes(**Genes.unflatten(gene_dict))
        hash_ = hash(new_dna.to_ck_string())
        if hash_ not in hashes:
            hashes.add(hash_)
            new_dataset.append(new_dna)

    print(f"Length of test geneset dataset: {len(new_dataset)}")
    print(f"duplicates removed: {len(base_dataset) - len(new_dataset)}")
    return new_dataset
import numpy as np 
from PIL import Image
from pathlib import Path 
from tqdm import tqdm
import insightface
from insightface.utils import face_align
from thirdparty.recognition.arcface_torch.backbones import get_model
import torch as th 
import torch.backends as backends
import itertools as it 
import warnings 
import torchvision.transforms as VT
from typing import Literal
from facenet_pytorch import InceptionResnetV1


if th.cuda.is_available():
    DEVICE = "cuda"
elif backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


class FaceDetector:
    def __init__(self):
        self.insight_face_app = insightface.app.FaceAnalysis(name="antelopev2", allowed_modules=["detection"])
        self.insight_face_app.prepare(ctx_id = 0, det_thresh = 0.1)

    def __call__(self, img: np.ndarray):
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        face = self.insight_face_app.get(img)[0]
        aimg = face_align.norm_crop(img, landmark=face.kps, image_size=112)
        return aimg


class FeatureExtractor(th.nn.Module):
    def __init__(self, model, layer_names):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.layer_names = layer_names
        self.outputs = []

        # Register hooks
        for layer_name in self.layer_names:
            layer = dict([*self.model.named_modules()])[layer_name]
            layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.outputs.append(output)

    def forward(self, x):
        self.outputs = []  # Reset outputs before each forward pass
        final = self.model(x)
        return [f for f in list(self.outputs) + [final]]
    
    def batch_forward(self, x, batch_size):
        with th.no_grad():
            partials = batch_predict(self, x, batch_size=batch_size)
        n_outputs = len(self.layer_names) + 1
        batches_per_layer = [list() for _ in range(n_outputs)]
        for batch_output in partials:
            for i in range(n_outputs):
                batches_per_layer[i].append(batch_output[i])
        return [th.concatenate(l) for l in batches_per_layer]
    
def get_earlier_layers_features(model, layers_to_hook, images: np.ndarray, device: str=DEVICE) -> list[np.ndarray]:
    transform = VT.Compose([
        VT.ToTensor(),
        VT.Resize((112, 112)),
        VT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    data = th.stack([transform(img) for img in tqdm(images)])
    data = data.to(device)
    model = model.to(device)

    fe = FeatureExtractor(model, layers_to_hook).eval()
    with th.no_grad():
        features: list[th.Tensor] = fe.batch_forward(data, 32)
    return [f.view(*f.size()[:2], -1).mean(-1).detach().cpu().numpy() for f in features]

def save_features_from_layers(model, layers_to_hook, X, output_dir: Path, model_name: str) -> None: 
    features = get_earlier_layers_features(model, layers_to_hook, X)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_names = ["embeddings_" + "_".join(l.split(".")) for l in layers_to_hook + [model_name]] 
    for feat, fname in zip(features, layer_names):
        np.save(output_dir / fname, feat)


def batch_predict(model, input, batch_size):
    partial_output = []
    total_batches = len(input) // batch_size
    for batch_start in tqdm(range(0, len(input), batch_size), total=total_batches):
        batch = input[batch_start: batch_start + batch_size]
        with th.no_grad():
            partial_output.append(model(batch))
    return partial_output
    
    
def allign_faces(images: np.ndarray, detector: FaceDetector, labels: list[str], paths: list[Path] | None=None) -> tuple[np.ndarray["X", float], np.ndarray["y", str]]:
    def handle_no_face_detected(img: np.ndarray, i: int) -> np.ndarray | None:
        try:
            return detector(img)
        except IndexError:
            if paths is not None:
                warnings.warn(f"No face detected in {paths[i]}")
            else:
                warnings.warn(f"No face detected in image {i}")
            return None
    
    alligned_images = [handle_no_face_detected(img, i) for i, img in tqdm(list(enumerate(images)))]
    filtered_images, filtered_labels = zip(*[(img, label) for img, label in zip(alligned_images, labels) if img is not None])
    if paths is not None:
        filtered_paths = [path for path, img in zip(paths, alligned_images) if img is not None]
        return np.array(filtered_images), np.array(filtered_labels), filtered_paths
    
    return np.array(filtered_images), np.array(filtered_labels)


def get_embeddings(images: np.ndarray, model: th.nn.Module, batch_size: int=32, device: str=DEVICE, disable_tqdm: bool = False) -> np.ndarray:
    transform = VT.Compose([
        VT.ToTensor(),
        VT.Resize((112, 112)),
        VT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    data = th.stack([transform(img) for img in tqdm(images, disable=disable_tqdm)])
    model = model.to(device)

    embedddings = []
    for i in tqdm(range(0, len(images), batch_size), disable=disable_tqdm):
        batch = data[i:i+batch_size].to(device)
        embeddings = model(batch).detach()
        embedddings.append(embeddings)  

    embeddings = th.cat(embedddings)
    return embeddings.detach().cpu().numpy()
    

def get_dataset(root_dir: Path) -> tuple[np.ndarray["X", float], np.ndarray["y", str], list[Path]]:
    extensions = ["png", "jpg", "jpeg"]
    paths = list(it.chain.from_iterable([root_dir.rglob(f"*.{ext}") for ext in extensions]))
    paths = sorted(paths)
    images = [np.array(Image.open(img_path)) for img_path in tqdm(paths)]
    labels = [img_path.relative_to(root_dir).parent for img_path in paths]
    return np.array(images), np.array(labels), paths

def load_insightface_model(weights_path: str) -> th.nn.Module:
    resnet_version = weights_path.split("_")[-1].split(".")[0]
    model = get_model(resnet_version)
    model.load_state_dict(th.load(weights_path, map_location=th.device('cpu')))
    model = model.eval()
    return model


def load_facenet_model(training_dataset: Literal['casia-webface', 'vggface2']) -> th.nn.Module:
    resnet = InceptionResnetV1(pretrained=training_dataset, device="cpu").eval()
    return resnet
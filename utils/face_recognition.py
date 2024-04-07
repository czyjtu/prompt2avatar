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


def get_embeddings(images: np.ndarray, model: th.nn.Module, batch_size: int=32, device: str=DEVICE) -> np.ndarray:
    transform = VT.Compose([
        VT.ToTensor(),
        VT.Resize((112, 112)),
        VT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    data = th.stack([transform(img) for img in tqdm(images)])
    model = model.to(device)

    embedddings = []
    for i in tqdm(range(0, images.shape[0], batch_size)):
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

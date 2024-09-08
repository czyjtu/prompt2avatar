import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.training import get_clip_embeddings 
import numpy as np 
import click 
import warnings 

import utils.face_recognition as ufr

import numpy as np

# project dependencies
from deepface.commons import image_utils
from deepface.modules import modeling, detection, preprocessing
from deepface.models.FacialRecognition import FacialRecognition
from sklearn.preprocessing import normalize 


def represent(
        img_path,
        model,
        enforce_detection: bool = False,
        detector_backend: str = "opencv",
        align: bool = True,
        expand_percentage: int = 0,
        normalization: str = "base",
        anti_spoofing: bool = False,
        max_faces = None,
    ):
        """
        Represent facial images as multi-dimensional vector embeddings.

        Args:
            img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
                or a base64 encoded image. If the source image contains multiple faces, the result will
                include information for each detected face.

            model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
                OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet

            enforce_detection (boolean): If no face is detected in an image, raise an exception.
                Default is True. Set to False to avoid the exception for low-resolution images.

            detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
                'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'.

            align (boolean): Perform alignment based on the eye positions.

            expand_percentage (int): expand detected facial area with a percentage (default is 0).

            normalization (string): Normalize the input image before feeding it to the model.
                Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

            anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

            max_faces (int): Set a limit on the number of faces to be processed (default is None).

        Returns:
            results (List[Dict[str, Any]]): A list of dictionaries, each containing the
                following fields:

            - embedding (List[float]): Multidimensional vector representing facial features.
                The number of dimensions varies based on the reference model
                (e.g., FaceNet returns 128 dimensions, VGG-Face returns 4096 dimensions).
            - facial_area (dict): Detected facial area by face detection in dictionary format.
                Contains 'x' and 'y' as the left-corner point, and 'w' and 'h'
                as the width and height. If `detector_backend` is set to 'skip', it represents
                the full image area and is nonsensical.
            - face_confidence (float): Confidence score of face detection. If `detector_backend` is set
                to 'skip', the confidence will be 0 and is nonsensical.
        """
        resp_objs = []

        # ---------------------------------
        # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
        target_size = model.input_shape
        if detector_backend != "skip":
            img_objs = detection.extract_faces(
                img_path=img_path,
                detector_backend=detector_backend,
                grayscale=False,
                enforce_detection=enforce_detection,
                align=align,
                expand_percentage=expand_percentage,
                anti_spoofing=anti_spoofing,
                max_faces=max_faces,
            )
        else:  # skip
            # Try load. If load error, will raise exception internal
            img, _ = image_utils.load_image(img_path)

            if len(img.shape) != 3:
                raise ValueError(f"Input img must be 3 dimensional but it is {img.shape}")

            # make dummy region and confidence to keep compatibility with `extract_faces`
            img_objs = [
                {
                    "face": img,
                    "facial_area": {"x": 0, "y": 0, "w": img.shape[0], "h": img.shape[1]},
                    "confidence": 0,
                }
            ]
        # ---------------------------------

        if max_faces is not None and max_faces < len(img_objs):
            # sort as largest facial areas come first
            img_objs = sorted(
                img_objs,
                key=lambda img_obj: img_obj["facial_area"]["w"] * img_obj["facial_area"]["h"],
                reverse=True,
            )
            # discard rest of the items
            img_objs = img_objs[0:max_faces]

        for img_obj in img_objs:
            if anti_spoofing is True and img_obj.get("is_real", True) is False:
                raise ValueError("Spoof detected in the given image.")
            img = img_obj["face"]

            # rgb to bgr
            img = img[:, :, ::-1]

            region = img_obj["facial_area"]
            confidence = img_obj["confidence"]

            # resize to expected shape of ml model
            img = preprocessing.resize_image(
                img=img,
                # thanks to DeepId (!)
                target_size=(target_size[1], target_size[0]),
            )

            # custom normalization
            img = preprocessing.normalize_input(img=img, normalization=normalization)

            embedding = model.forward(img)

            resp_objs.append(
                {
                    "embedding": embedding,
                    "facial_area": region,
                    "face_confidence": confidence,
                }
            )

        return resp_objs

@click.command()
@click.option(
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, exists=False),
    help="Output directory to generate embeddings into"
)
@click.option(
    "--dataset_dir",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    help="Data directory containing the dataset. The dataset should be structured as follows: <datadir>/<label>/*/<image>.<ext>"
)
@click.option(
    "--weights",
    default=None,
    type=click.Path(dir_okay=True, file_okay=True, exists=True),
    help="if insightface is used - Path to the model weights file, or directory containing the model weights files. The model weights should be named as follows: <model_type>_<resnet_version>.pth"
)
@click.option(
    "--training_dataset",
    default=None,
    type=click.Choice(["casia-webface", "vggface2"]),
    help="Training dataset to use for facenet model"
)
@click.option(
    "--model_type",
    type=click.Choice(["insightface", "facenet"]),
    help="Type of the model to use for embeddings generation"
)
@click.option(
    "--batch_size",
    default=32,
    type=int,
    help="Batch size to use for embeddings generation"
)
def main(out: str, dataset_dir: str, weights: str | None, model_type: str, training_dataset: str | None, batch_size: int):
    if model_type == "facenet":
        raise NotImplementedError("Facenet model is not implemented")
        # if training_dataset is None:
        #     models_paths = ["casia-webface", "vggface2"]
        #     print(f"Will evaluate facenet pretrained on {models_paths}")
        # if weights is not None:
        #     warnings.warn("Weights are ignored for facenet model")

    elif model_type == "insightface":
        if weights is None:
            raise ValueError("InsightFace model requires weights directory to be specified")
        
        if (weights_path := Path(weights)).is_dir():
            models_paths = [str(p) for p in weights_path.rglob("*_*.pth")]
        else:
            models_paths = [str(weights_path)]
        print(f"Found {len(models_paths)} inisghtface models: {models_paths}")


    print(f"Input dataset: {dataset_dir}")
    out_dir = Path(out)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"out dir: {out_dir}")
    out_dir_alligned_images = out_dir / "alligned_images.npy"
    out_dir_alligned_labels = out_dir / "alligned_labels.npy"
    out_dir_alligned_paths = out_dir / "alligned_paths.npy"
    out_dir_embeddings = str(out_dir / "embeddings_{model_weights}.npy")

    if out_dir_alligned_images.exists():
        print("Loading alligned images...")
        alligned_images = np.load(out_dir_alligned_images)
    else:
        print("Loading dataset...")
        dataset_path = Path(dataset_dir)
        images, labels, paths = ufr.get_dataset(dataset_path)
        print("Labels head:", labels[:5])

        print("Alligning faces...")
        detector = ufr.FaceDetector()
        alligned_images, alligned_labels, alligned_paths = ufr.allign_faces(images, detector, list(labels), paths)
        alligned_paths = np.array([str(path) for path in alligned_paths])

        print("Saving alligned images...")
        np.save(out_dir_alligned_images, alligned_images)
        np.save(out_dir_alligned_labels, alligned_labels)
        np.save(out_dir_alligned_paths, alligned_paths)
        # dnas = [open(path.replace("faces", "dna").replace("png", "txt"), "r").read() for path in alligned_paths]
        # np.save(out_dir / "dna.npy", dnas)

    # for model_path in models_paths:
    #     print(f"Loading model {model_path}...")
    #     if model_type == "facenet":
    #         raise NotImplementedError("Facenet model is not implemented")
    #     elif model_type == "insightface":
    #         model = ufr.load_insightface_model(model_path)

    #     m = dict(model.named_modules())
    #     layers_to_hook = []#[list(k for k in m.keys() if f"layer{i}" in k)[-1] for i in range(1, 5)]
    #     print(f"Getting embeddings from layers {layers_to_hook}...")
    #     model_weights_fname = model_path.split("/")[-1].split(".")[0]
    #     ufr.save_features_from_layers(model, layers_to_hook, alligned_images, out_dir, model_weights_fname)
    
    # print("clip embeddings")
    # import os 
    # os.environ['CURL_CA_BUNDLE'] = ''

    # import torch
    # clip_embeddings = get_clip_embeddings(alligned_images, device="mps", dtype=torch.float16).detach().cpu().numpy()
    # np.save(out_dir / "clip_embeddings.npy", clip_embeddings)

    print("deepface")

    model_deepface = modeling.build_model(
        task="facial_recognition", model_name="DeepFace"
    )
    embeddings = []
    from tqdm import tqdm 
    for f in tqdm(alligned_images, total=len(alligned_images)):
        emb = normalize(np.array(represent(f[[2, 1, 0]], model_deepface)[0]["embedding"]).reshape(-1, 1))
        embeddings.append(emb)
    embeddings = np.array(embeddings)
    np.save(out_dir / "deepface_embeddings.npy", embeddings)

        # embeddings = ufr.get_embeddings(alligned_images, model, batch_size=batch_size)
        # print(f"Saving embeddings with shape {embeddings.shape} to ", out_dir_embeddings)
        # np.save(out_dir_embeddings.format(model_weights=model_weights_fname), embeddings)


if __name__ == "__main__":
    main()

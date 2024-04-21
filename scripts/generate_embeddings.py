import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np 
import click 
import warnings 

import utils.face_recognition as ufr


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

    for model_path in models_paths:
        print(f"Loading model {model_path}...")
        if model_type == "facenet":
            raise NotImplementedError("Facenet model is not implemented")
        elif model_type == "insightface":
            model = ufr.load_insightface_model(model_path)
        print("Getting embeddings...")
        embeddings = ufr.get_embeddings(alligned_images, model, batch_size=batch_size)
        print("Saving embeddings...")
        model_weights_fname = model_path.split("/")[-1].split(".")[0]
        np.save(out_dir_embeddings.format(model_weights=model_weights_fname), embeddings)


if __name__ == "__main__":
    main()

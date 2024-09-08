from deepface import DeepFace
import numpy as np 


imgs = np.load("data/dataset_2023_03_26_embeddings/alligned_images.npy")

emb = DeepFace.represent(imgs[0][[2, 1, 0]], model_name="DeepFace")
print(emb)
import os 
from pathlib import Path
import sys 
sys.path.append(Path(__file__).resolve().parent.parent.__str__())

from utils.training import get_clip_embeddings
import numpy as np 
import torch 
import os
os.environ['CURL_CA_BUNDLE'] = ''
faces = np.load("/Users/maciej.czyjt/priv-dev/prompt2avatar/data/easyv2_geneset_embeddings/alligned_images.npy")
emb = get_clip_embeddings(faces, "mps", batch_size=32, dtype=torch.float16)



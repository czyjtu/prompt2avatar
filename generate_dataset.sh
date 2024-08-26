#!bin/bash
python scripts/generate_embeddings.py --dataset_dir "/Users/czyjtu/dev/prompt2avatar/experiments/medium_geneset_embeddings/predicted_faces_mlp/predicted_dnas_mlp" --out "/Users/czyjtu/dev/prompt2avatar/experiments/medium_geneset_embeddings/predicted_faces_mlp/predicted_dnas_mlp_embeddings" --model_type "insightface" --weights "./thirdparty/recognition/weights/arcface_r100.pth"

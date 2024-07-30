#!bin/bash
python scripts/generate_embeddings.py --dataset_dir "/Users/czyjtu/dev/prompt2avatar/data/dnas_dataset_2024_04_07_genesets/medium_geneset" --out "/Users/czyjtu/dev/prompt2avatar/data/medium_geneset_embeddings" --model_type "insightface" --weights "./thirdparty/recognition/weights/arcface_r100.pth"

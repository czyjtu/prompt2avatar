#!bin/bash
# python scripts/generate_embeddings.py --dataset_dir "/Users/czyjtu/dev/prompt2avatar/data/celeba_sample_augmentation/predicted_dnas_mlp_celeba" \
# --out "/Users/czyjtu/dev/prompt2avatar/data/celeba_sample_augmentation/predicted_dnas_mlp_celeba_embeddings" --model_type "insightface" --weights "./thirdparty/recognition/weights/arcface_r100.pth"

# python scripts/resize_images.py --in_dir "/Users/czyjtu/dev/prompt2avatar/data/dnas_dataset_2024_04_07_genesets/medium_geneset_v2/faces/dna" \
#     --out "/Users/czyjtu/dev/prompt2avatar/data/dnas_dataset_2024_04_07_genesets/medium_geneset_v2"

python scripts/generate_embeddings.py --dataset_dir "/Users/czyjtu/dev/prompt2avatar/data/dnas_dataset_2024_04_07_genesets/medium_geneset_v2/faces_resized" \
--out "/Users/czyjtu/dev/prompt2avatar/data/medium_geneset_v2_embeddings" --model_type "insightface" --weights "./thirdparty/recognition/weights/arcface_r100.pth"
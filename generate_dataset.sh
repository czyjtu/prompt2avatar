#!bin/bash
python scripts/generate_embeddings.py --dataset_dir "/Users/czyjtu/dev/prompt2avatar/data/genes_grid/gene_forehead_brow_height.forehead_brow_height_pos_skin_color[1]" \
--out "/Users/czyjtu/dev/prompt2avatar/data/genes_grid/gene_forehead_brow_height.forehead_brow_height_pos_skin_color[1]_embeddings" --model_type "insightface" --weights "./thirdparty/recognition/weights/arcface_r100.pth"

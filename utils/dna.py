from __future__ import annotations
from copy import deepcopy
import re
from pydantic import BaseModel,  NonNegativeInt
from typing import Literal
import numpy as np 
from collections import defaultdict

class Genes(BaseModel):
    type: Literal["female", "male"] = "female"
    hair_color: tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt, NonNegativeInt]
    skin_color: tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt, NonNegativeInt]
    eye_color: tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt, NonNegativeInt]
    gene_chin_forward: dict[str, NonNegativeInt]
    gene_chin_height: dict[str, NonNegativeInt]
    gene_chin_width: dict[str, NonNegativeInt]
    gene_eye_angle: dict[str, NonNegativeInt]
    gene_eye_depth: dict[str, NonNegativeInt]
    gene_eye_height: dict[str, NonNegativeInt]
    gene_eye_distance: dict[str, NonNegativeInt]
    gene_eye_shut: dict[str, NonNegativeInt]
    gene_forehead_angle: dict[str, NonNegativeInt]
    gene_forehead_brow_height: dict[str, NonNegativeInt]
    gene_forehead_roundness: dict[str, NonNegativeInt]
    gene_forehead_width: dict[str, NonNegativeInt]
    gene_forehead_height: dict[str, NonNegativeInt]
    gene_head_height: dict[str, NonNegativeInt]
    gene_head_width: dict[str, NonNegativeInt]
    gene_head_profile: dict[str, NonNegativeInt]
    gene_head_top_height: dict[str, NonNegativeInt]
    gene_head_top_width: dict[str, NonNegativeInt]
    gene_jaw_angle: dict[str, NonNegativeInt]
    gene_jaw_forward: dict[str, NonNegativeInt]
    gene_jaw_height: dict[str, NonNegativeInt]
    gene_jaw_width: dict[str, NonNegativeInt]
    gene_mouth_corner_depth: dict[str, NonNegativeInt]
    gene_mouth_corner_height: dict[str, NonNegativeInt]
    gene_mouth_forward: dict[str, NonNegativeInt]
    gene_mouth_height: dict[str, NonNegativeInt]
    gene_mouth_width: dict[str, NonNegativeInt]
    gene_mouth_upper_lip_size: dict[str, NonNegativeInt]
    gene_mouth_lower_lip_size: dict[str, NonNegativeInt]
    gene_mouth_open: dict[str, NonNegativeInt]
    gene_neck_length: dict[str, NonNegativeInt]
    gene_neck_width: dict[str, NonNegativeInt]
    gene_bs_cheek_forward: dict[str, NonNegativeInt]
    gene_bs_cheek_height: dict[str, NonNegativeInt]
    gene_bs_cheek_width: dict[str, NonNegativeInt]
    gene_bs_ear_angle: dict[str, NonNegativeInt]
    gene_bs_ear_inner_shape: dict[str, NonNegativeInt]
    gene_bs_ear_bend: dict[str, NonNegativeInt]
    gene_bs_ear_outward: dict[str, NonNegativeInt]
    gene_bs_ear_size: dict[str, NonNegativeInt]
    gene_bs_eye_corner_depth: dict[str, NonNegativeInt]
    gene_bs_eye_fold_shape: dict[str, NonNegativeInt]
    gene_bs_eye_size: dict[str, NonNegativeInt]
    gene_bs_eye_upper_lid_size: dict[str, NonNegativeInt]
    gene_bs_forehead_brow_curve: dict[str, NonNegativeInt]
    gene_bs_forehead_brow_forward: dict[str, NonNegativeInt]
    gene_bs_forehead_brow_inner_height: dict[str, NonNegativeInt]
    gene_bs_forehead_brow_outer_height: dict[str, NonNegativeInt]
    gene_bs_forehead_brow_width: dict[str, NonNegativeInt]
    gene_bs_jaw_def: dict[str, NonNegativeInt]
    gene_bs_mouth_lower_lip_def: dict[str, NonNegativeInt]
    gene_bs_mouth_lower_lip_full: dict[str, NonNegativeInt]
    gene_bs_mouth_lower_lip_pad: dict[str, NonNegativeInt]
    gene_bs_mouth_lower_lip_width: dict[str, NonNegativeInt]
    gene_bs_mouth_philtrum_def: dict[str, NonNegativeInt]
    gene_bs_mouth_philtrum_shape: dict[str, NonNegativeInt]
    gene_bs_mouth_philtrum_width: dict[str, NonNegativeInt]
    gene_bs_mouth_upper_lip_def: dict[str, NonNegativeInt]
    gene_bs_mouth_upper_lip_full: dict[str, NonNegativeInt]
    gene_bs_mouth_upper_lip_profile: dict[str, NonNegativeInt]
    gene_bs_mouth_upper_lip_width: dict[str, NonNegativeInt]
    gene_bs_nose_forward: dict[str, NonNegativeInt]
    gene_bs_nose_height: dict[str, NonNegativeInt]
    gene_bs_nose_length: dict[str, NonNegativeInt]
    gene_bs_nose_nostril_height: dict[str, NonNegativeInt]
    gene_bs_nose_nostril_width: dict[str, NonNegativeInt]
    gene_bs_nose_profile: dict[str, NonNegativeInt]
    gene_bs_nose_ridge_angle: dict[str, NonNegativeInt]
    gene_bs_nose_ridge_width: dict[str, NonNegativeInt]
    gene_bs_nose_size: dict[str, NonNegativeInt]
    gene_bs_nose_tip_angle: dict[str, NonNegativeInt]
    gene_bs_nose_tip_forward: dict[str, NonNegativeInt]
    gene_bs_nose_tip_width: dict[str, NonNegativeInt]
    face_detail_cheek_def: dict[str, NonNegativeInt]
    face_detail_cheek_fat: dict[str, NonNegativeInt]
    face_detail_chin_cleft: dict[str, NonNegativeInt]
    face_detail_chin_def: dict[str, NonNegativeInt]
    face_detail_eye_lower_lid_def: dict[str, NonNegativeInt]
    face_detail_eye_socket: dict[str, NonNegativeInt]
    face_detail_nasolabial: dict[str, NonNegativeInt]
    face_detail_nose_ridge_def: dict[str, NonNegativeInt]
    face_detail_nose_tip_def: dict[str, NonNegativeInt]
    face_detail_temple_def: dict[str, NonNegativeInt]
    expression_brow_wrinkles: dict[str, NonNegativeInt]
    expression_eye_wrinkles: dict[str, NonNegativeInt]
    expression_forehead_wrinkles: dict[str, NonNegativeInt]
    expression_other: dict[str, NonNegativeInt]
    complexion: dict[str, NonNegativeInt]
    gene_height: dict[str, NonNegativeInt]
    gene_bs_body_type: dict[str, NonNegativeInt]
    gene_bs_body_shape: dict[str, NonNegativeInt]
    gene_bs_bust: dict[str, NonNegativeInt]
    gene_age: dict[str, NonNegativeInt]
    gene_eyebrows_shape: dict[str, NonNegativeInt]
    gene_eyebrows_fullness: dict[str, NonNegativeInt]
    gene_body_hair: dict[str, NonNegativeInt]
    gene_hair_type: dict[str, NonNegativeInt]
    gene_baldness: dict[str, NonNegativeInt]
    eye_accessory: dict[str, NonNegativeInt]
    teeth_accessory: dict[str, NonNegativeInt]
    eyelashes_accessory: dict[str, NonNegativeInt]
    clothes: dict[str, NonNegativeInt]


    @staticmethod
    def from_ck_string(string: str) -> Genes:
        gender_pattern = r'type=(\w+)'
        genes_pattern = r'genes=(.*)'
        key_val_pattern = r'(\w+)\s*=\s*{([^}]+)}'
        list_pattern = r'\s*(\d+) (\d+) (\d+) (\d+)\s*'
        string_pattern = r'"([^"]+)" (\d+)'

        gender = re.findall(gender_pattern, string)[0]
        genes_all = re.findall(genes_pattern, string, re.DOTALL)[0]

        genes_dict = {"type": gender}
        for key, value in re.findall(key_val_pattern, genes_all):
            values = re.findall(list_pattern, value)
            if not values:
                values = {key: val for key, val in re.findall(string_pattern, value)}
            else:
                values = tuple(map(int, values[0]))
            genes_dict[key] = values
        return Genes(**genes_dict)
    
    @staticmethod
    def unflatten(flattened_genes: dict[str, int]) -> dict[str, int | dict[str, int] | tuple[int, int, int, int]]:
        genes_dict = {}
        for key, val in flattened_genes.items():
            if "[" in key:
                key, index = key.split("[")
                index = int(index[:-1])
                if key not in genes_dict:
                    genes_dict[key] = [0, 0, 0, 0]
                genes_dict[key][index] = val
            elif "." in key:
                key, sub_key = key.split(".")
                if key not in genes_dict:
                    genes_dict[key] = {}
                genes_dict[key][sub_key] = val
            else:
                genes_dict[key] = val
        return genes_dict
    
    @staticmethod
    def from_array(array: np.ndarray, keys: list[str], template_flattened: dict[str, int] | None = None) -> Genes:
        genes_dict = Genes.unflatten({key: val for key, val in zip(keys, array)})
        template_dict = Genes.unflatten(template_flattened or {})
        template_dict.update(genes_dict)
        return Genes(**template_dict)

    def to_ck_string(self) -> str:
        ck_string = "ruler_designer={\ntype=" + self.type + "\nid=0\ngenes={"
        for key, value in self.dict(exclude={"type"}).items():
            if isinstance(value, tuple):
                value_str = f" {value[0]} {value[1]} {value[2]} {value[3]} "
            else:
                value_str = "".join([f' "{sub_key}" {sub_val} ' for sub_key, sub_val in value.items()])
                if len(value) == 1:
                    value_str *= 2

            ck_string += f"\t{key}={{{value_str}}}\n"
        ck_string += "}"
        return ck_string
    
    def flatten(self) -> dict[str, int]:
        result = {}
        for gene_name, gene_val in self.dict(exclude={"type"}).items():
            if isinstance(gene_val, tuple):
                for i, val in enumerate(gene_val):
                    result[f"{gene_name}[{i}]"] = val
            else:
                for sub_gene_name, sub_gene_val in gene_val.items():
                    # TODO: for now we treat ..._neg as ..._pos 
                    # as we didn't spot any difference between faces generated using both of them
                    if sub_gene_name.endswith("_neg"):
                        sub_gene_name = sub_gene_name[:-4] + "_pos"
                    name = f"{gene_name}.{sub_gene_name}"
                    result[name] = sub_gene_val
        return result

    def asarray(self, keys: list[str] | None) -> np.ndarray:
        flatten = self.flatten()
        keys = keys or sorted(flatten.keys())
        sordered_values = [flatten[key] for key in keys]
        return np.array(sordered_values, dtype=np.uint16)


def create_template_from_genes(genes_list: list[Genes]):
    flattened_dna = [dna.flatten() for dna in genes_list]
    temmplate = defaultdict(list)
    for dna in flattened_dna:
        for key, val in dna.items():
            temmplate[key].append(val)
    avg_template = {key: np.mean(val) for key, val in temmplate.items()}
    return avg_template
        
    

if __name__ == "__main__":
    fpath = "data/dataset_2023_03_26/male/arabic/dna/20230327_015121422378.txt"
    with open(fpath, "r") as f:
        string = f.read()
        genes = Genes.from_ck_string(string)
        print(genes)

    print(genes.to_ck_string())
    print(genes.flatten())
    print(genes.asarray())

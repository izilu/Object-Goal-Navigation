import os
from typing import Union
from constants import coco_categories

import numpy as np
import torch
from torchtext.vocab import GloVe

def get_glove_embedding(model_type: str, dim: int, cls_num: int, text: Union[str, list], save_path: str):
    glove = GloVe(name=model_type, dim=dim)
    text_embedding = []
    for item in text:
        emb = glove.get_vecs_by_tokens(item, True)
        if float(emb.sum()) == 0:
            if ' ' in item:
                split_list = item.split(" ")
            else:
                split_list = item.split("-")
            split_emb = [glove.get_vecs_by_tokens(i, True) for i in split_list]
            emb = torch.stack(split_emb).mean(dim=0)
        text_embedding.append(emb)
    text_embedding = torch.stack(text_embedding)
    print(text_embedding)
    np.save(os.path.join(save_path, f"glove{model_type.replace('/', '').replace('-', '')}_textfeatures_{cls_num}cls.npy"), text_embedding.to("cpu").numpy())

if __name__ == "__main__":
    model = "6B"
    save_path = "./object_embeddings"
    cls_num = 15

    get_glove_embedding(model, dim=300, cls_num=cls_num, text=list(coco_categories.keys()), save_path=save_path)

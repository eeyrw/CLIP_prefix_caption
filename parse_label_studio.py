import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"F:/COCO_DS/hunk_split_{clip_model_name}_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('100.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    embeddingIndex = 0
    for i in tqdm(range(len(data))):
        d=dict()
        imageUrl = data[i]["data"]["captioning"]
        anno_items = data[i]["annotations"][0]["result"]
        if data[i]["annotations"][0]["was_cancelled"]:
            continue
        d['caption']=''
        for anno_item in anno_items:
            if anno_item['from_name']=='caption':
                for caption in anno_item['value']['text']:
                    d['caption'] = caption
                    d["clip_embedding"] = embeddingIndex
                    all_captions.append(d)

        relPath = imageUrl.split("=", 1)[1]
        absPath = os.path.join('F:/NSFW_DS',relPath)
        image = io.imread(absPath)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        
        all_embeddings.append(prefix)
        embeddingIndex = embeddingIndex+1
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-L/14@336px", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32','ViT-L/14@336px'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))

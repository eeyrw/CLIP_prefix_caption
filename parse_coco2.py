import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader

class CocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.imageList)

    def __getitem__(self, item: int):
        return self.imageprocessor(Image.open(self.imageList[item]))

    def __init__(self,clip_model_type,device):
        self.device = device
        with open('F:/COCO_DS/train_caption.json', 'r') as f:
            data = json.load(f)
        print("%0d captions is loaded from json " % len(data))
        self.imageList=[]
        for d in tqdm(data):
            img_id = d["image_id"]
            filename = f"F:/COCO_DS/train2014/COCO_train2014_{int(img_id):012d}.jpg"
            if not os.path.isfile(filename):
                filename = f"F:/COCO_DS/val2014/COCO_val2014_{int(img_id):012d}.jpg"
            self.imageList.append(filename)
        _, self.imageprocessor = clip.load(clip_model_type, device=device, jit=False)


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"F:/COCO_DS/2oscar_split_{clip_model_name}_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    parse_dataloader = DataLoader(CocoDataset(clip_model_type,device), batch_size=100, shuffle=False, drop_last=False,num_workers=4)
    all_embeddings = []
    all_captions = []
    i = 0
    d = dict()
    for images in tqdm(parse_dataloader):
        images = images.to(device)
        with torch.no_grad():
            prefix_pr = clip_model.encode_image(images).cpu()
        for prefix in prefix_pr:
            d["clip_embedding"] = i
            i = i+1
            all_embeddings.append(prefix)
            all_captions.append(d)
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
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))

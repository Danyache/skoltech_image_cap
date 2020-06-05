import argparse
import os
import torch
import tqdm
import numpy as np
import pandas as pd

import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from sklearn.metrics import roc_auc_score
from pathlib import Path
curr_path = Path(os.getcwd())

splits_path = os.path.join(str(curr_path.parent), 'Transformer-Based-Generation/splits')
output_path = os.path.join(str(curr_path.parent), 'Transformer-Based-Generation/output_cxr')
print(splits_path, output_path)

import torchvision
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from torchvision import transforms

from data import ImageDataset
from model.inception import inception_v3_base
from collections import OrderedDict

class DenseNet121(nn.Module):
	"""Model modified.
	The architecture of our model is the same as standard DenseNet121
	except the classifier layer which has an additional sigmoid function.
	"""
	def __init__(self, out_size):
		super(DenseNet121, self).__init__()
		self.densenet121 = torchvision.models.densenet121(pretrained=True)
		num_ftrs = self.densenet121.classifier.in_features
		self.densenet121.classifier = nn.Sequential(
		    nn.Linear(num_ftrs, out_size),
		    nn.Sigmoid()
		)

	def forward(self, x):
		x = self.densenet121(x)
		return x

good_layers = ["densenet121.features.conv0.weight", "densenet121.features.norm0.weight", "densenet121.features.norm0.bias", "densenet121.features.norm0.running_mean", "densenet121.features.norm0.running_var", "densenet121.features.transition1.norm.weight", "densenet121.features.transition1.norm.bias", "densenet121.features.transition1.norm.running_mean", "densenet121.features.transition1.norm.running_var", "densenet121.features.transition1.conv.weight", "densenet121.features.transition2.norm.weight", "densenet121.features.transition2.norm.bias", "densenet121.features.transition2.norm.running_mean", "densenet121.features.transition2.norm.running_var", "densenet121.features.transition2.conv.weight", "densenet121.features.transition3.norm.weight", "densenet121.features.transition3.norm.bias", "densenet121.features.transition3.norm.running_mean", "densenet121.features.transition3.norm.running_var", "densenet121.features.transition3.conv.weight", "densenet121.features.norm5.weight", "densenet121.features.norm5.bias", "densenet121.features.norm5.running_mean", "densenet121.features.norm5.running_var", "densenet121.classifier.0.weight", "densenet121.classifier.0.bias"]

def split_file(split):
    return os.path.join(splits_path, f'mysplit_{split}_images.txt')


def read_split_image_ids_and_paths(split):
    split_df = pd.read_csv(split_file(split), sep=' ', header=None)
    return np.array(split_df.iloc[:,1]), np.array(split_df.iloc[:,0])

def main(args):
    image_ids, image_paths = read_split_image_ids_and_paths('train')
    image_ids2, image_paths2 = read_split_image_ids_and_paths('test')
    image_ids3, image_paths3 = read_split_image_ids_and_paths('test')
    
    image_ids = list(image_ids)
    image_ids2 = list(image_ids2)
    image_ids3 = list(image_ids3)
    
    image_paths = list(image_paths)
    image_paths2 = list(image_paths2)
    image_paths3 = list(image_paths3)
    
    image_ids.extend(image_ids2)
    image_ids.extend(image_ids3)
    
    image_paths.extend(image_paths2)
    image_paths.extend(image_paths3)
    
    image_ids = np.array(image_ids)
    image_paths = np.array(image_paths)
    
    image_paths = [image_path for image_path in image_paths]
    features_dir = os.path.join(output_path, f'{args.split}-features-grid')

    os.makedirs(features_dir, exist_ok=True)

    inception = DenseNet121(8).cuda() # inception_v3_base(pretrained=True)
    state_dict = torch.load("/raid/data/cxr14-2/DenseNet121_aug4_pretrain_WeightBelow1_1_0.829766922537.pkl")
    new_state_dict = OrderedDict()

    for s, v in state_dict.items():

        if 'module.' in s:
            s = s.replace('module.', '')

        if s not in good_layers:
            s = '.'.join(s.split('.')[:-2]) + '.'.join(s.split('.')[-2:])

        new_state_dict[s] = v
    inception.load_state_dict(new_state_dict)
    
    inception.eval()
    inception.to(args.device)
    
    dense_m = inception._modules['densenet121']

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(image_ids, image_paths, transform=transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         pin_memory=args.device.type == 'cuda',
                                         shuffle=False)

    with torch.no_grad():
        for imgs, ids in tqdm.tqdm(loader):
            # outs = inception(imgs.to(args.device)).permute(0, 2, 3, 1).view(-1, 64, 2048)
            imgs = imgs.to(args.device)
            for f in dense_m.features:
                imgs = f(imgs)
            outs = imgs.permute(0, 2, 3, 1).view(-1, 49, 1024)
            for out, id in zip(outs, ids):
                out = out.cpu().numpy()
                id = str(id)
                np.save(os.path.join(features_dir, id), out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MS-COCO captions pre-processing.')

    parser.add_argument('--ms-coco-dir',
                        help='MS-COCO data directory.')
    parser.add_argument('--split', choices=['train', 'valid', 'test'],
                        help="Data split ('train', 'valid' or 'test').")
    parser.add_argument('--output-dir', default='output_cxr',
                        help='Output directory.')
    parser.add_argument('--device', default='cuda', type=torch.device,
                        help="Device to use ('cpu', 'cuda', ...).")
    parser.add_argument('--batch-size', default=8, type=int,
                        help="Image batch size.")
    parser.add_argument('--num-workers', default=0, type=int,
                        help="Number of data loader workers.")

    main(parser.parse_args())

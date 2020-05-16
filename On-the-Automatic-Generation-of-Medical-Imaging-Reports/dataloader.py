import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os, re
import nltk
from collections import Counter
# from build_vocab import Vocabulary, build_vocab


import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os, re
import nltk
import pickle
import argparse
from collections import Counter

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
import pandas as pd


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(captions, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    for i in range(len(captions)):
        caption = captions[i]
#         print(caption)
        for j in range(len(caption)):
#             print(caption[j])
            tokens = nltk.tokenize.word_tokenize(caption[j].lower())
            counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(captions)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

asd = pd.read_pickle('/home/dchesakov/NLMCXR_data/all_reports_df.pkl')
# vocab = build_vocab(asd.processed_captions.values, 8)
asd = asd[ asd['processed_findings'].notnull() ]
vocab = build_vocab(asd.processed_findings.values, 8)

def load_label_list(file_list):
    labels = []
    filename_list = []
    with open(file_list, 'r') as f:
        for line in f:
            items = line.split()
            image_name = items[0]
            label = items[1:]
            label = [int(i) for i in label]
            image_name = '{}.png'.format(image_name)
            filename_list.append(image_name)
            labels.append(label)
    return filename_list, labels


class iuxray(Dataset):
    def __init__(self, data_file="train_data.txt", vocab = vocab, transform=None):
        # self.root_dir = root_dir
        # self.tsv_path = tsv_path
        # self.image_path = image_path
        
        # tsv_file = os.path.join(self.root_dir, self.tsv_path)
        self.asd = pd.read_pickle('/home/dchesakov/NLMCXR_data/all_reports_df.pkl')
        self.asd = self.asd[ self.asd['processed_findings'].notnull() ]
        # self.captions = create_captions(tsv_file)
        self.captions_dict = {}
        
        for image, caption in zip(self.asd.images, self.asd.processed_findings):
            self.captions_dict[image+'.png'] = caption

        self.vocab = vocab

        # self.data_file = pd.read_csv(tsv_file, delimiter='\t',encoding='utf-8')
        self.transform = transform

        # self.all_imgs, _ = load_label_list(f"/home/dchesakov/ImageCapProject/{data_file}")
        self.all_imgs = asd.images.values

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        # print(idx)

        img_name = self.all_imgs[idx] + '.png'
        image = Image.open(f'/home/dchesakov/ImageCapProject/NLMCXR_png/{img_name}')

        # img_name = os.path.join(self.root_dir, self.image_path, self.data_file.iloc[idx, 0])
        # image = Image.open(img_name)
        
        if self.transform is not None:
            image = self.transform(image)
        
#         try:
#             caption = new_caption(self.captions[img_name])
#         except:
#             caption = new_caption('normal. ')
        caption = self.captions_dict[img_name]

        # print("---")
        # print(caption)
        # print("----")

        sentences = []

        for i in range(len(caption)):
            tokens = nltk.tokenize.word_tokenize(str(caption[i]).lower())
            sentence = []
            sentence.append(self.vocab('<start>'))
            sentence.extend([self.vocab(token) for token in tokens])
            sentence.append(self.vocab('<end>'))
            # print([self.vocab.idx2word[k] for k in sentence])
            sentences.append(sentence)

        # print([self.vocab.idx2word[sentences[0][k]] for k in sentences[0]]) 
            
        max_sent_len = max([len(sentences[i]) for i in range(len(sentences))])
        
        for i in range(len(sentences)):
            if len(sentences[i]) < max_sent_len:
                sentences[i] = sentences[i] + (max_sent_len - len(sentences[i]))* [self.vocab('<pad>')]
                
        target = torch.Tensor(sentences)

        return image, target, len(sentences), max_sent_len


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption, no_of_sent, max_sent_len).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption, no_of_sent, max_sent_len). 
            - image: torch tensor of shape (3, crop_size, crop_size).
            - caption: torch tensor of shape (no_of_sent, max_sent_len); variable length.
            - no_of_sent: number of sentences in the caption
            - max_sent_len: maximum length of a sentence in the caption
    Returns:
        images: torch tensor of shape (batch_size, 3, crop_size, crop_size).
        targets: torch tensor of shape (batch_size, max_no_of_sent, padded_max_sent_len).
        prob: torch tensor of shape (batch_size, max_no_of_sent)
    """
    # Sort a data list by caption length (descending order).
#     data.sort(key=lambda x: len(x[1]), reverse=True)
    
    images, captions, len_sentences, max_sent_len = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    
    targets = torch.zeros(len(captions), max(len_sentences), max(max_sent_len)).long()
    prob = torch.zeros(len(captions), max(len_sentences)).long()
    
    for i, cap in enumerate(captions):
        for j, sent in enumerate(cap):
            targets[i, j, :len(sent)] = sent[:] 
            prob[i, j] = 1
        # stop after the last sentence
        # prob[i, j] = 0
      
    return images, targets, prob

def get_loader(data_file, transform, batch_size, shuffle, num_workers, vocab = vocab):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # dataset
    data = iuxray(data_file=data_file,
             vocab = vocab,
             transform = transform)
    
    # Data loader for dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, resize_length, resize_width).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset = data, 
                                              batch_size = batch_size,
                                              shuffle = shuffle,
                                              num_workers = num_workers,
                                              collate_fn = collate_fn)

    return data_loader, data.vocab
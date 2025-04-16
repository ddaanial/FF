
import os
from pathlib import Path
import random
import json
from PIL import Image
import torch
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import lightning as L
from fuseformer_poetry.data.preprocess_data import create_random_shape_with_random_motion 
from fuseformer_poetry.data.preprocess_data import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip




def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index


class CusDataset(Dataset):
    def __init__(self, args: dict, split='train', video_names=None):
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']
        self.size = self.w, self.h = (args['w'], args['h'])
        if video_names is None:
            parent_path = Path().resolve()
            vid_lst_prefix = os.path.join(parent_path, self.args['data_root'])
            video_names = [os.path.join(vid_lst_prefix, name) for name in os.listdir(vid_lst_prefix)]
        self.video_names = video_names
        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]
        all_frames = [
                os.path.join(video_name, name) 
                for name in sorted(os.listdir(video_name)) 
                if not name.startswith('.')
            ]

        all_masks = create_random_shape_with_random_motion(
            len(all_frames), imageHeight=self.h, imageWidth=self.w)
        
        if self.split == 'test':
            ref_index = [i for i in range(self.args['num_test_frames'])]
        else:
            ref_index = get_ref_index(len(all_frames), self.sample_length)
        # read video frames
        frames = []
        masks = []
        for idx in ref_index:
            img = Image.open(all_frames[idx]).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            masks.append(all_masks[idx])
        if self.split == 'train':
            frames = GroupRandomHorizontalFlip()(frames)
        # To tensors
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        return frame_tensors, mask_tensors

class CusDataModule(L.LightningDataModule):
    def __init__(self, arg):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        parent_path = Path().resolve()
        vid_lst_prefix = os.path.join(parent_path, self.hparams.arg['data_loader']['data_root'])
        video_names = [
                os.path.join(vid_lst_prefix, name) 
                for name in os.listdir(vid_lst_prefix) 
                if not name.startswith('.')
            ]

        num_videos = len(video_names)
        # Define split sizes (80% train, 10% val, 10% test)
        train_size = int(0.8 * num_videos)
        val_size = int(0.1 * num_videos)
        test_size = num_videos - train_size - val_size
        # Generate reproducible permutation of indices
        indices = torch.randperm(
            num_videos, generator=torch.Generator().manual_seed(self.hparams.arg['seed'])
        ).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]


        # Create video lists for each split
        train_videos = [video_names[i] for i in train_indices]
        val_videos = [video_names[i] for i in val_indices]
        test_videos = [video_names[i] for i in test_indices]

        # Initialize datasets with appropriate split values

        if stage == 'fit':
            self.train_dataset = CusDataset(self.hparams.arg['data_loader'], split='train', video_names = train_videos)
            self.val_dataset = CusDataset(self.hparams.arg['data_loader'], split='val', video_names = val_videos)
        if stage == 'test':
            self.test_dataset = CusDataset(self.hparams.arg['data_loader'], split='test', video_names = test_videos)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.arg['trainer']['train_val_batch_size'],
            shuffle=True,
            num_workers= self.hparams.arg['trainer']['num_workers']
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.arg['trainer']['train_val_batch_size'],
            shuffle=False,
            num_workers= self.hparams.arg['trainer']['num_workers']
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.arg['trainer']['test_batch_size'],
            shuffle=False,
            num_workers= self.hparams.arg['trainer']['num_workers']
        )
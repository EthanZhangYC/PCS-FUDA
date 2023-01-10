import cv2
import glob
import torch.utils.data as data
import numpy as np
import torch
from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import math
import re
import pandas as pd

stat_dict_ourdata = {
 'ct':{'1001': (-0.412, 0.345), '1002': (0.048, 0.364), '1004': (0.062, 0.417), '1005': (0.104, 0.383), 
 '1006': (-0.065, 0.281), '1007': (0.168, 0.351), '1009': (-0.147, 0.308), '1010': (-0.172, 0.404), 
 '1011': (-0.088, 0.35), '1012': (0.096, 0.381),'1013': (0.026, 0.418), '1015': (-0.109, 0.463),
 '1016': (-0.407, 0.354), '1017': (0.087, 0.464), '1018': (0.189, 0.374), '1020': (-0.406, 0.348),
 '1008': (0.464,0.269), '1019': (0.313,0.227), '1014': (0.239,0.215), '1003': (0.254,0.192)},
 'mr':{'1001': (-0.796, 0.174), '1002': (-0.613, 0.291), '1003': (-0.586, 0.513), '1004': (-0.773, 0.367),
 '1005': (-0.732, 0.417), '1006': (-0.58, 0.488), '1008': (-0.696, 0.451), '1010': (-0.609, 0.48),
 '1011': (-0.599, 0.5), '1012': (-0.726, 0.228), '1013': (-0.563, 0.341), '1014': (-0.625, 0.272),
 '1015': (-0.497, 0.526), '1016': (-0.748, 0.373), '1017': (-0.775, 0.339), '1020': (-0.691, 0.421),
 '1007': (0.205, 0.234), '1009': (0.206, 0.166), '1018': (0.173, 0.235), '1019': (0.146, 0.208)},                  
 'cyc_ct':{'1001': (-0.317, 0.387), '1002': (0.05, 0.402), '1004': (0.084, 0.441), '1005': (0.104, 0.4),
 '1006': (0.002, 0.353), '1007': (0.132, 0.382), '1009': (-0.079, 0.326), '1010': (-0.088, 0.428),
 '1011': (-0.012, 0.382), '1012': (0.104, 0.397), '1013': (0.067, 0.433), '1015': (-0.032, 0.465),
 '1016': (-0.328, 0.377), '1017': (0.105, 0.472), '1018': (0.183, 0.405), '1020': (-0.314, 0.388)},
 'cyc_mr':{'1001': (-0.701, 0.147), '1002': (-0.588, 0.29), '1003': (-0.513, 0.475), '1004': (-0.69, 0.305),
 '1005': (-0.667, 0.374), '1006': (-0.51, 0.458), '1008': (-0.627, 0.401), '1010': (-0.566, 0.452),
 '1011': (-0.542, 0.462), '1012': (-0.695, 0.178), '1013': (-0.481, 0.375), '1014': (-0.54, 0.281),
 '1015': (-0.448, 0.516), '1016': (-0.714, 0.326), '1017': (-0.734, 0.285), '1020': (-0.643, 0.376)},
 'fake_ct':{'1001': (-0.451, 0.259), '1002': (-0.13, 0.346), '1003': (-0.204, 0.445), '1004': (-0.425, 0.402),
 '1005': (-0.4, 0.394), '1006': (-0.182, 0.457), '1008': (-0.32, 0.455), '1010': (-0.249, 0.452),
 '1011': (-0.294, 0.47), '1012': (-0.259, 0.293), '1013': (-0.152, 0.331), '1014': (-0.143, 0.331),
 '1015': (-0.138, 0.478), '1016': (-0.429, 0.418), '1017': (-0.457, 0.367), '1020': (-0.348, 0.435),
 '1007': (-0.027, 0.403), '1009': (-0.23, 0.801), '1018': (-0.137, 0.428), '1019': (-0.236, 0.409)},
 'fake_mr':{'1001': (-0.706, 0.289), '1002': (-0.687, 0.43), '1004': (-0.654, 0.459), '1005': (-0.646, 0.425),
 '1006': (-0.695, 0.298), '1007': (-0.615, 0.413), '1009': (-0.73, 0.273), '1010': (-0.742, 0.332),
 '1011': (-0.724, 0.356), '1012': (-0.655, 0.414), '1013': (-0.684, 0.414), '1015': (-0.714, 0.375),
 '1016': (-0.71, 0.284), '1017': (-0.629, 0.411), '1018': (-0.599, 0.438), '1020': (-0.712, 0.284),
 '1003': (-0.766, 0.259), '1008': (-0.672, 0.389), '1014': (-0.767, 0.294), '1019': (-0.79, 0.258)}}

vol_stat = {
'mr':{'1001': 64, '1002': 40, '1003': 80, '1004': 88,
 '1005': 80, '1006': 56, '1007': 130, '1008': 104,
 '1009': 120, '1010': 88, '1011': 96, '1012': 72,
 '1013': 56, '1014': 64, '1015': 64, '1016': 72,
 '1017': 80, '1018': 130, '1019': 130, '1020': 80},
'ct':{'1001': 144, '1002': 88, '1003': 256, '1004': 128,
 '1005': 112, '1006': 88, '1007': 104, '1008': 256,
 '1009': 112, '1010': 104, '1011': 144, '1012': 112,
 '1013': 144, '1014': 256, '1015': 96, '1016': 152,
 '1017': 88, '1018': 128, '1019': 256, '1020': 144}
}

MODAL2REAL = {
    "fake_ct": "mr",
    "fake_mr": "ct",
    "cyc_ct": "ct",
    "cyc_mr": "mr",
    "org_mr": "mr",
    "org_ct": "ct",
    "ct": "ct",
    "mr": "mr"
}

def sort_vol_slice(path):
    vol = re.findall('[a-z]+_([0-9]+)_.+?\.npy', path.split('/')[-1])[0]
    # vol = re.findall('[a-z]+_([0-9]+)_.+?\.npy', path.split('/')[-1])[0]
    slice_ = re.findall('[a-z]+_[0-9]+_([0-9]+).+', path.split('/')[-1])[0]
    return int(vol)*1000+int(slice_)

def create_label_list(image_list):
    label_list = np.array([np.load(re.sub('\.npy','_label.npy',img_path)).astype(np.uint8)[:32,:32] for img_path in image_list])
    return label_list

class WHS_dataset(data.Dataset):
    def __init__(self, data_dir, transforms=None, zscore=True, labeled=False):
        print(data_dir)
        # import pdb;pdb.set_trace()
        if isinstance(data_dir, list):
            self.raw_data = []
            for dd in data_dir:
                self.raw_data += [f for f in glob.glob(dd+'/*') if '_label.npy' not in f]
        elif '.txt' in data_dir:
            with open(data_dir, 'r') as fp:
                self.raw_data = [f.strip().replace('/home1/ziyuan/UDA/data/data_leuda_sifa_new/mr','/home/yichen/ProtoUDA/data_mmwhs/mr') for f in fp.readlines()]
        else:
            self.raw_data = [f for f in glob.glob(data_dir+'/*') if '_label.npy' not in f]
        # self.raw_data.sort() # check
        self.imgs = self.raw_data = sorted(self.raw_data, key = lambda x: sort_vol_slice(x))
        if labeled:
            self.labeled = True
            self.labels = create_label_list(self.imgs)
        else:
            self.labeled = False
        # print(self.raw_data[:10])
        self.transform = transforms
        self.idx_mapping = {i: i for i in range(len(self.raw_data))}
        self.zscore=zscore
        
    def update_idx_mapping(self, idx_mapping=None):
        """
        For targetlike_dataloaders, first half of the data and the second half is of different modality but have same origin, (e.g. `fake_mr` and `ct`)
        and in order to train structure loss, we need to align slices from both modalities in pairs.
        So we disable the default shuffle in outside dataloader, and instead use `idx_mapping` to shuffle slices. 
        Here only the first half of data is shuffled, then copy and shift the assignment to the second half. 
        """
        if idx_mapping is not None:
            self.idx_mapping = idx_mapping
        else:
            num_label_slices = len(self.raw_data)
            self.idx_mapping = {i: i_ for i, i_ in zip(range(num_label_slices), np.random.permutation(range(num_label_slices)))}

    def __getitem__(self, idx):
        # keep consistent with fake ct 1014, which has 65 slices.
        idx =  self.idx_mapping[idx]
        img_path = self.raw_data[idx]
        img = np.load(img_path)
        
        img_modal = re.findall('(.+?)_[0-9]+.+', img_path.split('/')[-1])[0]
        img_vol = re.findall('.+_([0-9]+)_[0-9]+.+', img_path.split('/')[-1])[0]
        img_slice = re.findall('.+_[0-9]+_([0-9]+).+', img_path.split('/')[-1])[0]
        img_name = img_path.split('/')[-1]
        
        if self.zscore:
            mean, std = stat_dict_ourdata[img_modal][img_vol]
            img = (img-mean)/std
        else:
            img = img * 2 - 1.0
            
        if self.labeled:
            gt_path = re.sub('\.npy','_label.npy',img_path)            
            gt = np.load(gt_path)
            if self.transform:
                seq = iaa.Affine(scale=(0.9, 1.1), rotate=(-10, 10))
                seq_det = seq.to_deterministic()
                gt = gt.astype(np.uint8)
                segmap = SegmentationMapsOnImage(gt, shape=gt.shape)
                img, segmaps_aug = seq_det(image=img, segmentation_maps=segmap)
                gt = segmaps_aug.arr.squeeze()
            gt = gt[:32,:32].astype(np.uint8)

        else:
            gt=0
            # gt_path = re.sub('\.npy','_label.npy',img_path)            
            # gt = np.load(gt_path.replace('_','_labeled'))
            # if self.transform:
            #     seq = iaa.Affine(scale=(0.9, 1.1), rotate=(-10, 10))
            #     seq_det = seq.to_deterministic()
            #     gt = gt.astype(np.uint8)
            #     segmap = SegmentationMapsOnImage(gt, shape=gt.shape)
            #     gt = segmaps_aug.arr.squeeze().astype(np.uint8)

        real_modal = MODAL2REAL[img_modal]
        vol_len = vol_stat[real_modal][img_vol]
        
        img = img[:32,:32].astype(float)
        

        return idx, img[np.newaxis,...], gt
        # return {"image": img[np.newaxis,...], "label": gt.astype(np.uint8),'vol_id':img_vol, 'slice_id':img_slice, "idx":idx}

    def __len__(self):
        return len(self.raw_data)
    
if __name__ == '__main__':
    # data_dir = '/data1/ziyuan/fangcheng/MMWHS_v6.1/ct_npy_new/ct_train'
    # train_set = WHS_dataset('/data1/ziyuan/fangcheng/data_shumeng/mr/mr_like_test/org_mr/', supervised=True, zscore=False)
    # print(train_set)
    # data_list = '/home1/ziyuan/UDA/data/mmwhs_bare/split_csv/barely_labeled.csv'
    # data_list = '/home1/ziyuan/UDA/data/mmwhs_bare/split_csv/barely_sourcelike_train.csv'
    # data_list = '/home1/ziyuan/UDA/data/mmwhs_bare/split_csv/label4_org.csv'
    data_dir = '/home/ziyuan/UDA/data/mmwhs/ct/ct_train'
    
    train_set = WHS_dataset(data_dir=data_dir, transforms=True)
    for i in range(20):
        sample = train_set[i]
        print(sample['image'].shape,sample['label'].shape, sample['vol_id'], sample['slice_id'])


import os
from torch.utils.data.dataset import Dataset
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from einops import rearrange, repeat


class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        # self.scale = args.scale_factor
        self.patchsize = args.patch_for_train
        self.task = args.task
        self.dataset_dir = args.path_for_train + args.task + '/'
        self.data_list = args.data_list_for_train

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                repeat_num = 1 if tmp_list[index] in ['kitchen', 'museum', 'vinyl'] else 3
                for _ in range(repeat_num):
                    self.file_list.append(data_name + '/' + tmp_list[index])

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index % self.item_num]]
        with h5py.File(file_name[0], 'r') as hf:
            LF = np.array(hf.get('LF'))  # LF image
            if self.task == 'Disp':
                DispGT = np.array(hf.get('dispGT'))

        LF = torch.from_numpy(LF)
        if self.task == 'Disp':
            DispGT = torch.from_numpy(DispGT)

        LF_1, DispGT_1 = self.augmentation(LF, DispGT)
        LF_2, DispGT_2 = self.augmentation(LF, DispGT)
        return LF_1, DispGT_1, LF_2, DispGT_2

    def augmentation(self, LF, DispGT):
        """ Data Augmentation """
        # LF, DispGT = refocus_augmentation(LF, DispGT)
        # LF, DispGT = scale_augmentation(LF, DispGT, self.patchsize)
        LF, DispGT = random_crop(LF, DispGT, self.patchsize)
        LF, DispGT = orientation_augmentation(LF, DispGT)
        LF = color_shuffle(LF)

        random_sheard_value = random.randint(-3, 3)
        LF = SheardEPI(LF, random_sheard_value)
        DispGT[0:1, :, :] = DispGT[0:1, :, :] - random_sheard_value
        return LF, DispGT

    def __len__(self):
        if self.task == 'Disp':
            return self.item_num * 50


def color_shuffle(LF):
    ''' random color shuffling '''
    color = [0, 1, 2]
    random.shuffle(color)
    LF = LF[color, :, :, :, :]
    return LF


def orientation_augmentation(data, dispGT):
    if random.random() < 0.5:  # flip along W-V direction
        data = torch.flip(data, dims=[2, 4])
        dispGT = torch.flip(dispGT, dims=[2])
    if random.random() < 0.5:  # flip along W-V direction
        data = torch.flip(data, dims=[1, 3])
        dispGT = torch.flip(dispGT, dims=[1])
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.permute(0, 2, 1, 4, 3)
        dispGT = dispGT.permute(0, 2, 1)

    return data, dispGT


def random_crop(lf, dispGT, patchsize):
    C, U, V, H, W = lf.size()
    h_idx = np.random.randint(0, H - patchsize)
    w_idx = np.random.randint(0, W - patchsize)
    out_lf = lf[:, :, :, h_idx : h_idx + patchsize, w_idx : w_idx + patchsize]
    out_disp = dispGT[:, h_idx : h_idx + patchsize, w_idx : w_idx + patchsize]
    return out_lf, out_disp


def refocus_augmentation(lf, dispGT):
    C, U, V, H, W = lf.size()
    min_d = torch.floor(torch.min(dispGT))
    max_d = torch.ceil(torch.max(dispGT))
    dd = np.random.randint(-5 - min_d, 5 - max_d)
    out_dispGT = dispGT + dd
    for u in range(U):
        for v in range(V):
            dh = int(dd * (u - U // 2))
            dw = int(dd * (v - V // 2))
            lf[:, u, v, :, :] = torch.roll(lf[:, u, v, :, :], shifts=int(dd * (u - U // 2)), dims=1)
            lf[:, u, v, :, :] = torch.roll(lf[:, u, v, :, :], shifts=int(dd * (v - V // 2)), dims=2)
            if dh < 0:
                lf[:, u, v, dh::, :] = 0
            elif dh > 0:
                lf[:, u, v, 0:dh, :] = 0
            if dw < 0:
                lf[:, u, v, :, dw::] = 0
            elif dw > 0:
                lf[:, u, v, :, 0:dw] = 0
    lf = lf[:, :, :, abs(dh):H-abs(dh), abs(dw):W-abs(dw)]
    out_dispGT = out_dispGT[:, abs(dh):H-abs(dh), abs(dw):W-abs(dw)]
    return lf, out_dispGT


def scale_augmentation(lf, dispGT, patchsize):
    scale = np.random.randint(1, 4)
    out_lf = lf[:, :, :, 0::scale, 0::scale]
    out_disp = dispGT[:, 0::scale, 0::scale]
    out_disp = out_disp / scale

    return out_lf, out_disp


def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    data_list = args.data_list_for_test

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name)
        length_of_tests += len(test_Dataset)

        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name):
        super(TestSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in_for_test
        self.task = args.task
        self.dataset_dir = args.path_for_test + args.task + '/'
        self.data_list = [data_name]

        self.file_list = []
        tmp_list = os.listdir(self.dataset_dir + data_name)
        for index, _ in enumerate(tmp_list):
            tmp_list[index] = data_name + '/' + tmp_list[index]
        self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            LF = np.array(hf.get('LF'))  # LF image
            if self.task == 'Disp':
                DispGT = np.array(hf.get('dispGT'))

        LF = torch.from_numpy(LF)
        if self.task == 'Disp':
            DispGT = torch.from_numpy(DispGT)
            DispGT = DispGT[0:1, :, :]

        # ShearedValue = +4
        # LF = SheardEPI(LF, ShearedValue)
        # DispGT = DispGT - ShearedValue

        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return [LF, DispGT, LF_name]

    def __len__(self):
        return self.item_num


def SheardEPI(LF, shift):
    [_, U, V, _, _] = LF.size()
    # LF = rearrange(LF, 'c u v h w -> c u v h w', u=angRes, v=angRes)
    for u in range(U):
        for v in range(V):
            t1 = int(shift * (u - U // 2))
            t2 = int(shift * (v - V // 2))
            LF[:, u, v, :, :] = torch.roll(LF[:, u, v, :, :], shifts=int(shift * (u - U // 2)), dims=1)
            LF[:, u, v, :, :] = torch.roll(LF[:, u, v, :, :], shifts=int(shift * (v - V // 2)), dims=2)
            if t1 < 0:
                LF[0, u, v, t1::, :] = 0
            elif t1 > 0:
                LF[0, u, v, 0:t1, :] = 0
            if t2 < 0:
                LF[0, u, v, :, t2::] = 0
            elif t2 > 0:
                LF[0, u, v, :, 0:t2] = 0
    return LF
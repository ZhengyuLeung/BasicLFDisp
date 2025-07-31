import argparse
import os
import h5py
import numpy as np
from pathlib import Path
from utils.func_pfm import read_pfm
import imageio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Disp', help='SSR, ASR')
    parser.add_argument('--data_for', type=str, default='training', help='')
    parser.add_argument('--src_data_path', type=str, default='./datasets/', help='')
    parser.add_argument('--save_data_path', type=str, default='./', help='')
    return parser.parse_args()


def main(args):
    angRes = 9
    patch_size = 512
    stride = patch_size // 2

    ''' 建立保存路径 '''
    save_dir = Path(args.save_data_path + 'data_for_' + args.data_for)
    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir.joinpath(args.task)
    save_dir.mkdir(exist_ok=True)

    ''' 遍历光场数据集 '''
    src_datasets = os.listdir(args.src_data_path)
    src_datasets.sort()
    for index_dataset in range(len(src_datasets)):
        if src_datasets[index_dataset] not in ['HCI_new_Disp']:
            continue
        name_dataset = src_datasets[index_dataset]
        ''' 建立数据集对应的保存路径 '''
        sub_save_dir = save_dir.joinpath(name_dataset)
        sub_save_dir.mkdir(exist_ok=True)

        ''' 遍历数据集下的场景 '''
        src_sub_dataset = args.src_data_path + name_dataset + '/' + args.data_for + '/'
        files = os.listdir(src_sub_dataset)
        for file in files:
            print('Generating training data of Scene_%s in Dataset %s......\t' % (file, name_dataset))
            idx_scene_save = 0
            LF = np.zeros(shape=(angRes, angRes, 512, 512, 3), dtype=int)

            """ Read inputs """
            for i in range(angRes**2):
                temp = imageio.imread(args.src_data_path + name_dataset + '/' + args.data_for + '/' + file + '/input_Cam0%.2d.png' % i)
                LF[i // angRes, i - angRes * (i // angRes), :, :, :] = temp

            LF = (LF / 255.0).astype('float32')
            (U, V, H, W, _) = LF.shape

            dispGT = np.zeros(shape=(H, W, 2), dtype=float)
            dispGT[:, :, 0] = np.float32(read_pfm(args.src_data_path + name_dataset + '/' + args.data_for + '/' + file + '/gt_disp_lowres.pfm'))
            mask_rgb = imageio.v2.imread(args.src_data_path + name_dataset + '/' + args.data_for + '/' + file + '/valid_mask.png')
            dispGT[:, :, 1] = np.float32(mask_rgb[:, :, 1] > 0)
            dispGT = dispGT.astype('float32')

            for h in range(0, H - patch_size + 1, stride):
                for w in range(0, W - patch_size + 1, stride):
                    idx_scene_save = idx_scene_save + 1
                    patch_LF = np.zeros((U, V, patch_size, patch_size, 3), dtype='float32')

                    patch_LF[:, :, :, :, :] = LF[:, :, h: h + patch_size, w: w + patch_size, :]
                    patch_dispGT = dispGT[h: h + patch_size, w: w + patch_size, :]

                    # save
                    file_name = [str(sub_save_dir) + '/' + file + '.h5']
                    with h5py.File(file_name[0], 'w') as hf:
                        # 注：matlab在保存h5py文件，好像会倒序保存
                        # 为了和matlab生成的数据格式保持一致，这里也做了transpose
                        hf.create_dataset('LF', data=patch_LF.transpose((4, 0, 1, 2, 3)), dtype='float32')
                        hf.create_dataset('dispGT', data=patch_dispGT.transpose((2, 0, 1)), dtype='float32')
                        hf.close()
                        pass
                    pass
                pass

            print('%d training samples have been generated\n' % (idx_scene_save))

            pass
        pass

    pass


if __name__ == '__main__':
    args = parse_args()

    main(args)
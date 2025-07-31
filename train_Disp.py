from common import main
from option import args


args.patch_for_train = 42
args.patch_size_for_test = 64
args.stride_for_test = 32
args.minibatch_for_test = 1

if __name__ == '__main__':
    args.device = 'cuda:0'
    args.num_workers = 4
    args.angRes_in = 9
    args.angRes_in_for_test = 9
    args.data_list_for_train = ['HCI_new_Disp']
    args.data_list_for_test = ['HCI_new_Disp']

    args.model_name = 'EPIT_Disp'
    args.path_pre_pth = './pth/xxxx.pth'

    args.start_epoch = 0
    main(args)


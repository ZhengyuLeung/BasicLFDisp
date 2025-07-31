import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader
from collections import OrderedDict
import imageio
from utils.func_pfm import write_pfm


def main(args):
    ''' Create Dir for Save'''
    log_dir, checkpoints_dir, val_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA Training LOADING '''
    logger.log_string('\nLoad Training Dataset ...')
    train_Dataset = TrainSetDataLoader(args)
    logger.log_string("The number of training data is: %d" % len(train_Dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True,)

    ''' DATA Validation LOADING '''
    logger.log_string('\nLoad Validation Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    logger.log_string("The number of validation data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)


    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
        start_epoch = 0
        logger.log_string('Do not use pre-trained model!')
    else:
        try:
            ckpt_path = args.path_pre_pth
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            start_epoch = checkpoint['epoch']
            try:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = 'module.' + k  # add `module.`
                    new_state_dict[name] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
            except:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
        except:
            net = MODEL.get_model(args)
            net.apply(MODEL.weights_init)
            start_epoch = 0
            logger.log_string('No existing model, starting training from scratch...')
            pass
        pass
    start_epoch = args.start_epoch
    net = net.to(device)
    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    cudnn.benchmark = True

    ''' Print Parameters '''
    logger.log_string('PARAMETER ...')
    logger.log_string(args)


    ''' LOSS LOADING '''
    criterion = MODEL.get_loss(args).to(device)


    ''' Optimizer '''
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad == True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)


    ''' TRAINING & TEST '''
    logger.log_string('\nStart training...')
    for idx_epoch in range(start_epoch, args.epoch):
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))

        ''' Training '''
        loss_epoch_train = train(train_loader, device, net, criterion, optimizer)
        logger.log_string('The %dth Train, loss is: %.5f' % (idx_epoch + 1, loss_epoch_train))

        ''' Save PTH  '''
        if args.local_rank == 0:
            save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_%s_epoch_%02d_model.pth' % (args.model_name,
                                                                                           args.angRes_in,
                                                                                           args.angRes_in,
                                                                                           args.task,
                                                                                           idx_epoch + 1)
            state = {
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
            }
            torch.save(state, save_ckpt_path)
            logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))

        ''' Validation '''
        step = 5
        if idx_epoch%step == 0 or idx_epoch > 50:
            with torch.no_grad():
                ''' Create Excel for MSE_100/BP_07 '''
                excel_file = ExcelFile()

                MSE_100_list = []
                BP_07_list = []
                BP_03_list = []
                BP_01_list = []
                for index, test_name in enumerate(test_Names):
                    test_loader = test_Loaders[index]

                    epoch_dir = val_dir.joinpath('VAL_epoch_%02d' % (idx_epoch + 1))
                    epoch_dir.mkdir(exist_ok=True)
                    save_dir = epoch_dir.joinpath(test_name)
                    save_dir.mkdir(exist_ok=True)

                    MSE_100, BP_07, BP_03, BP_01 = test(args, test_name, test_loader, net, excel_file, save_dir)
                    excel_file.write_sheet(test_name, 'Average', 'MSE_100', MSE_100)
                    excel_file.write_sheet(test_name, 'Average', 'BP_07', BP_07)
                    excel_file.write_sheet(test_name, 'Average', 'BP_03', BP_03)
                    excel_file.write_sheet(test_name, 'Average', 'BP_01', BP_01)
                    excel_file.add_count(2)

                    MSE_100_list.append(MSE_100)
                    BP_07_list.append(BP_07)
                    BP_03_list.append(BP_03)
                    BP_01_list.append(BP_01)

                    logger.log_string('The %dth Test on %s, MSE_100/BP_07/BP_03/BP_01 is %.2f/%.4f/%.4f/%.4f' % (
                    idx_epoch + 1, test_name, MSE_100, BP_07, BP_03, BP_01))
                    pass
                MSE_100_mean = np.array(MSE_100_list).mean()
                BP_07_mean = np.array(BP_07_list).mean()
                BP_03_mean = np.array(BP_03_list).mean()
                BP_01_mean = np.array(BP_01_list).mean()
                excel_file.write_sheet('ALL', 'Average', 'MSE_100', MSE_100_mean)
                excel_file.write_sheet('ALL', 'Average', 'BP_07', BP_07_mean)
                logger.log_string('The mean MSE_100 on testsets is %.5f, mean BP_07 is %.4f, mean BP_03 is %.4f, '
                                  'mean BP_01 is %.4f' % (MSE_100_mean, BP_07_mean, BP_03_mean, BP_01_mean))
                excel_file.xlsx_file.save(str(epoch_dir) + '/evaluation_%s_%s.xlsx' % (args.task, args.model_name))
                pass
            pass

        ''' scheduler '''
        scheduler.step()
        pass
    pass


def train(train_loader, device, net, criterion, optimizer):
    ''' training one epoch '''
    loss_list = []

    for idx_iter, (LF_1, DispGT_1, LF_2, DispGT_2) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
        LF = torch.cat([LF_1, LF_2], dim=0)
        DispGT = torch.cat([DispGT_1, DispGT_2], dim=0)
        ''' degradation '''
        if args.task == 'Disp':
            [_, _, max_U, max_V, _, _] = LF.size()
            U_in, V_in = args.angRes_in, args.angRes_in
            LF_input = LF[:, 0:3, (max_U - U_in)//2:(max_U + U_in)//2, (max_V - V_in) // 2:(max_V + V_in) // 2, :, :]

            LF_target = DispGT
            info = None

        ''' super-resolve the degraded LF images'''
        LF_input = LF_input.to(device)      # low resolution
        LF_target = LF_target.to(device)    # high resolution
        # GPU_cost = torch.cuda.memory_allocated(0)
        net.train()
        LF_out = net(LF_input, info)
        # GPU_cost1 = (torch.cuda.memory_allocated(0) - GPU_cost) / 1024 / 1024 / 1024  # GB
        loss = criterion(LF_out, LF_target, info)

        ''' calculate loss and MSE_100/BP_07 '''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()

        loss_list.append(loss.data.cpu())
        pass

    loss_mean = float(np.array(loss_list).mean())
    return loss_mean


def test(args, test_name, test_loader, net, excel_file, save_dir=None):
    MSE_100_list = []
    BP_07_list = []
    BP_03_list = []
    BP_01_list = []

    for idx_iter, (LF, DispGT, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        if args.task == 'Disp':
            [_, _, max_U, max_V, _, _] = LF.size()
            U_in, V_in = args.angRes_in, args.angRes_in
            LF_input = LF[:, 0:3, (max_U - U_in) // 2:(max_U + U_in) // 2, (max_V - V_in) // 2:(max_V + V_in) // 2, :, :]

            LF_target = DispGT
            info = None

        ''' Crop LFs into Patches '''
        LF_divide_integrate_func = LF_divide_integrate(1, args.patch_size_for_test, args.stride_for_test)
        sub_LF_input = LF_divide_integrate_func.LFdivide(LF_input)


        ''' SR the Patches '''
        import time
        sub_LF_out = []
        for i in range(0, sub_LF_input.size(0), args.minibatch_for_test):
            tmp = sub_LF_input[i:min(i + args.minibatch_for_test, sub_LF_input.size(0)), :, :, :, :, :]
            with torch.no_grad():
                net.eval()
                torch.cuda.empty_cache()
                out = net(tmp.to(args.device), info)
                sub_LF_out.append(out['Disp'])
        sub_LF_out = torch.cat(sub_LF_out, dim=0)
        LF_out = LF_divide_integrate_func.LFintegrate(sub_LF_out).unsqueeze(0)
        LF_out = LF_out[:, :, 0:LF_target.size(-2), 0:LF_target.size(-1)].cpu().detach()

        ''' Calculate the MSE_100 & BP_07 '''
        mse_x100, bad_pixel_07, bad_pixel_03, bad_pixel_01 = cal_metrics(LF_target, LF_out, bd=15)
        excel_file.write_sheet(test_name, LF_name[0], 'MSE_100', mse_x100)
        excel_file.write_sheet(test_name, LF_name[0], 'BP_07', bad_pixel_07)
        excel_file.write_sheet(test_name, LF_name[0], 'BP_03', bad_pixel_03)
        excel_file.write_sheet(test_name, LF_name[0], 'BP_01', bad_pixel_01)
        excel_file.add_count(1)
        MSE_100_list.append(mse_x100)
        BP_07_list.append(bad_pixel_07)
        BP_03_list.append(bad_pixel_03)
        BP_01_list.append(bad_pixel_01)
        print('   %s: MSE_100/BP_07/BP_03/BP_01 is %.3f/%.4f/%.4f/%.4f' % (
            LF_name[0], mse_x100, bad_pixel_07, bad_pixel_03, bad_pixel_01))

        ''' Save Disparity PFM '''
        if save_dir is not None:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)

            path = str(save_dir_) + '/' + LF_name[0] + '.pfm'
            write_pfm(np.float32(LF_out.squeeze().data.cpu()), path)

            path = str(save_dir_) + '/' + LF_name[0] + '.bmp'
            plt.imsave(path, LF_out.squeeze().clamp(LF_target.min(), LF_target.max()), cmap=plt.cm.viridis)

            path = str(save_dir_) + '/' + LF_name[0] + '_error.bmp'
            error = (LF_target - LF_out)
            plt.imsave(path, error.squeeze().clamp(-0.2, 0.2), cmap=plt.cm.seismic)

            path = str(save_dir_) + '/' + LF_name[0] + '_BP07.bmp'
            error = torch.abs(LF_target - LF_out)
            error = torch.where(error >= 0.07, 0, 1).float()
            plt.imsave(path, error.squeeze().clamp(0.0, 1), cmap=plt.cm.RdYlGn, vmin=0.0, vmax=1.25)

            path = str(save_dir_) + '/' + LF_name[0] + '_BP03.bmp'
            error = torch.abs(LF_target - LF_out)
            error = torch.where(error >= 0.03, 0, 1).float()
            plt.imsave(path, error.squeeze().clamp(0.0, 1), cmap=plt.cm.RdYlGn, vmin=0.0, vmax=1.25)

            path = str(save_dir_) + '/' + LF_name[0] + '_BP01.bmp'
            error = torch.abs(LF_target - LF_out)
            error = torch.where(error >= 0.01, 0, 1).float()
            plt.imsave(path, error.squeeze().clamp(0.0, 1), cmap=plt.cm.RdYlGn, vmin=0.0, vmax=1.25)
            pass

    return [np.array(MSE_100_list).mean(), np.array(BP_07_list).mean(),
            np.array(BP_03_list).mean(), np.array(BP_01_list).mean()]


def main_test(args):
    # if args.model_name not in args.path_pre_pth:
    #     from pre_model_dict import pre_model_dict
    #     args.path_pre_pth = pre_model_dict[args.model_name + str(args.scale_factor)]
    ''' Create Dir for Save '''
    _, _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('TEST')
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA TEST LOADING '''
    print('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    print("The number of test data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    print('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    ''' Load Pre-Trained PTH '''
    checkpoint = torch.load(args.path_pre_pth, map_location='cpu')
    try:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = 'module.' + k  # add `module.`
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)
        print('Use pretrain model!')
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k] = v
        # load params
        net.load_state_dict(new_state_dict)
        print('Use pretrain model!')
        pass

    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    print('PARAMETER ...')
    print(args)

    ''' TEST on every dataset '''
    print('\nStart test...')
    with torch.no_grad():
        ''' Create Excel for MSE_100/BP_07 '''
        excel_file = ExcelFile()

        MSE_100_list = []
        BP_07_list = []
        BP_03_list = []
        BP_01_list = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            save_dir = result_dir.joinpath(test_name)
            save_dir.mkdir(exist_ok=True)

            MSE_100, BP_07, BP_03, BP_01 = test(args, test_name, test_loader, net, excel_file, save_dir)
            excel_file.write_sheet(test_name, 'Average', 'MSE_100', MSE_100)
            excel_file.write_sheet(test_name, 'Average', 'BP_07', BP_07)
            excel_file.write_sheet(test_name, 'Average', 'BP_03', BP_03)
            excel_file.write_sheet(test_name, 'Average', 'BP_01', BP_01)
            excel_file.add_count(2)

            MSE_100_list.append(MSE_100)
            BP_07_list.append(BP_07)
            BP_03_list.append(BP_03)
            BP_01_list.append(BP_01)
            print('Test on %s, MSE_100/BP_07/BP_03/BP_01 is %.2f/%.4f/%.4f/%.4f' % (test_name, MSE_100, BP_07, BP_03, BP_01))
            pass

        MSE_100_mean = np.array(MSE_100_list).mean()
        BP_07_mean = np.array(BP_07_list).mean()
        BP_03_mean = np.array(BP_03_list).mean()
        BP_01_mean = np.array(BP_01_list).mean()
        excel_file.write_sheet('ALL', 'Average', 'MSE_100', MSE_100_mean)
        excel_file.write_sheet('ALL', 'Average', 'BP_07', BP_07_mean)
        excel_file.write_sheet('ALL', 'Average', 'BP_03', BP_03_mean)
        excel_file.write_sheet('ALL', 'Average', 'BP_01', BP_01_mean)
        print('The mean MSE_100 on testsets is %.2f, mean BP_07 is %.4f, mean BP_03 is %.4f, mean BP_01 is %.4f' % (MSE_100_mean, BP_07_mean, BP_03_mean, BP_01_mean))
        excel_file.xlsx_file.save(str(result_dir) + '/evaluation_%s_%s.xlsx' % (args.task, args.model_name))
    pass


def main_inference(args):
    _, _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('INFERENCE')
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA TEST LOADING '''
    print('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    print("The number of test data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    print('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    ''' Load Pre-Trained PTH '''
    checkpoint = torch.load(args.path_pre_pth, map_location='cpu')
    try:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = 'module.' + k  # add `module.`
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)
        print('Use pretrain model!')
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k] = v
        # load params
        net.load_state_dict(new_state_dict)
        print('Use pretrain model!')
        pass

    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    print('PARAMETER ...')
    print(args)

    ''' TEST on every dataset '''
    print('\nStart test...')
    with torch.no_grad():
        ''' Create Excel for MSE_100/BP_07 '''
        excel_file = ExcelFile()
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            save_dir = result_dir.joinpath(test_name)
            save_dir.mkdir(exist_ok=True)

            MSE_100, BP_07, BP_03, BP_01 = test(args, test_name, test_loader, net, excel_file, save_dir)
            pass
    pass
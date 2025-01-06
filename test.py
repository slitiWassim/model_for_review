import pprint
import argparse
import tqdm
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import Datasets
from utils import train_util, log_util, anomaly_util
from config.defaults import _C as config, update_config
from models.wresnet1024_cattn_tsm import ASTNet as get_net1
from models.wresnet2048_multiscale_cattn_tsmplus_layer6 import ASTNet as get_net2
import time

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# --cfg experiments/sha/sha_wresnet.yaml --model-file output/shanghai/sha_wresnet/shanghai.pth GPUS [3]
# --cfg experiments/ped2/ped2_wresnet.yaml --model-file output/ped2/ped2_wresnet/ped2.pth GPUS [3]
def parse_args():
    parser = argparse.ArgumentParser(description='Test Anomaly Detection')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='config/shanghaitech_wresnet.yaml', type=str)
    parser.add_argument('--model-file', help='model parameters',
                        default='pretrained/shanghaitech.pth', type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(config, args)

    logger, final_output_dir, tb_log_dir = \
        log_util.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False       # TODO ? False
    config.freeze()

    gpus = [(config.GPUS[0])]
    # model = models.get_net(config)
    if config.DATASET.DATASET == "ped2":
        model = get_net1(config, pretrained=False)
    else:
        model = get_net1(config, pretrained=False)
    logger.info('Model: {}'.format(model.get_name()))
    model = nn.DataParallel(model, device_ids=gpus).cuda(device=gpus[0])
    logger.info('Epoch: '.format(args.model_file))

    
    # load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    test_dataset = eval('Datasets.get_test_data')(config)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    
    mat=anomaly_util.get_labels(config.DATASET.DATASET)
    psnr_list, fps = inference(config, test_loader, model,args)
    assert len(psnr_list) == len(mat), f'Ground truth has {len(mat)} videos, BUT got {len(psnr_list)} detected videos!'

    auc, fpr, tpr = anomaly_util.calculate_auc(config, psnr_list, mat)

    logger.info(f'AUC: {auc * 100:.1f}%  FPS : {fps} ')


def inference(config, data_loader, model,args,quit=False):
    loss_func_mse = nn.MSELoss(reduction='none')
    processing_time , frame_number = 0 , 0 
    
    model.eval()
    psnr_list = []
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df  # number of frames to process
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if(not(quit)):
              print('[{}/{}]'.format(i+1, len(data_loader)))
            psnr_video = []
            # compute the output
            video, video_name = train_util.decode_input(input=data, train=False)
            video = [frame.to(device=config.GPUS[0]) for frame in video]
            for f in tqdm.tqdm(range(len(video) - fp),disable=quit):
                inputs = video[f:f + fp]
                start_time = time.time()
                output = model(inputs)
                frame_number += 1
                if frame_number != 1 : 
                    processing_time += time.time() - start_time
                target = video[f + fp:f + fp + 1][0]

                # compute PSNR for each frame
                # https://github.com/cvlab-yonsei/MNAD/blob/d6d1e446e0ed80765b100d92e24f5ab472d27cc3/utils.py#L20
                mse_imgs = torch.mean(loss_func_mse((output[0] + 1) / 2, (target[0] + 1) / 2)).item()
                psnr = anomaly_util.psnr_park(mse_imgs)
                psnr_video.append(psnr)

            psnr_list.append(psnr_video)
    return psnr_list , ((frame_number-1)/processing_time)


if __name__ == '__main__':
    main()


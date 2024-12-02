import argparse
from deeplearning.utils.file_utils import gen_random_str
import torch
import numpy as np
import random
import os

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, required=False,
                        help='Random seed for reproducibility.')
    parser.add_argument('--model', default="unet_pcsdg_simple_SABA", required=False,
                        help='Model name.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--log_folder',default="/app/JiangJiaxuan/Segmentation/pixel-level/deeplearning/log/pcsdg_simple_SABA_Magrabia" ,required=False,
                        help='Log folder.')
    parser.add_argument('--tag', default="{}".format(gen_random_str()), required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[224, 224], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='batch size.')
    parser.add_argument('--initial_lr', type=float, default=1e-2, required=False,
                        help='initial learning rate.')
    parser.add_argument('--save_interval', type=int, default=25, required=False,
                        help='save_interval.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--no_shuffle', default=False, required=False, action='store_true',
                        help="No shuffle training set.")
    parser.add_argument('--num_threads', type=int, default=16, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root',default="/app/JiangJiaxuan/Segmentation/Hopkins/ccsdg/data/", required=False,
                        help='dataset root folder.')
    parser.add_argument('--tr_csv', nargs='+',
                        default=["/app/JiangJiaxuan/Segmentation/Hopkins/ccsdg/data/Magrabia_train.csv",
                                 "/app/JiangJiaxuan/Segmentation/Hopkins/ccsdg/data/Magrabia_test.csv"],required=False, help='training csv file.')
    parser.add_argument('--ts_csv', nargs='+',
                        default=["/app/JiangJiaxuan/Segmentation/Hopkins/ccsdg/data/MESSIDOR_Base3_train.csv",
                                 "/app/JiangJiaxuan/Segmentation/Hopkins/ccsdg/data/MESSIDOR_Base3_test.csv",
                                 "/app/JiangJiaxuan/Segmentation/Hopkins/ccsdg/data/MESSIDOR_Base2_train.csv",
                                 "/app/JiangJiaxuan/Segmentation/Hopkins/ccsdg/data/MESSIDOR_Base2_test.csv",
                                 "/app/JiangJiaxuan/Segmentation/Hopkins/ccsdg/data/MESSIDOR_Base1_train.csv",
                                 "/app/JiangJiaxuan/Segmentation/Hopkins/ccsdg/data/MESSIDOR_Base1_test.csv",
                                 "/app/JiangJiaxuan/Segmentation/Hopkins/ccsdg/data/REFUGE_Test.csv",
                                 "/app/JiangJiaxuan/Segmentation/Hopkins/ccsdg/data/REFUGE_Train.csv",
                                 "/app/JiangJiaxuan/Segmentation/Hopkins/ccsdg/data/REFUGE_Validation.csv"],required=False, help='test csv file.')
    parser.add_argument('--num_epochs', type=int, default=100, required=False,
                        help='num_epochs.')

    args = parser.parse_args()
    model_name = args.model
    now_seed = args.seed
    args.tag = f"Magrabia_{args.seed}"

    if model_name == 'unet':
        from deeplearning.training.train_nets.train_unet import train
    elif model_name == 'unet_pcsdg_simple_SABA':
        from deeplearning.training.train_nets.train_unet_pcsdg_simple_SABA import train
    else:
        print('No model named "{}"!'.format(model_name))
        return
    seed_torch(seed=now_seed)
    train(args)

if __name__ == '__main__':
    main()

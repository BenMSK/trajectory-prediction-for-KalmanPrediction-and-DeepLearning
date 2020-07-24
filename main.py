# -*- coding: utf-8 -*-
import argparse
import os
# import pickle
# import time

import torch

from utils import DataSet, PreprocessData
from torch.utils.data import DataLoader
from vanilla_gru import NNPred
from model import VanillaGRU
from kalman_model import KalmanModel
import matplotlib.pyplot as plt

CUDA = True
DEVICE = 'cuda:0'
TRAIN = False
TEST = True
DATASET = "Apol"# "Apol", "Lyft"

def main():
    parser = argparse.ArgumentParser()

    # Set a torch manual seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed of torch"
    )

    parser.add_argument(
        "--wandb",
        type=bool,
        default=False,
        help="Wandb trigger"
    )
    
    parser.add_argument(
        '--cuda', '-g', 
        action='store_true', 
        help='GPU option', 
        default=CUDA
    )

    parser.add_argument(
        '--device', '-d', 
        help='cuda device option', 
        default=DEVICE, 
        type=str
    )

    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default='../data/{}/'.format(DATASET),
        help="Data Directory"
    )

    parser.add_argument(
        "--obs_length",
        type=int,
        default=6,
        help="History length of the trajectory"
    )

    parser.add_argument(
        "--pred_length",
        type=int,
        default=10,
        help="predicted length of the trajectory"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="mini-batch size of Data"
    )

    parser.add_argument(
        "--train",
        type=bool,
        default=TRAIN,
        help="Activate the train mode. If not, the test mode is activated"
    )

    parser.add_argument(
        "--test",
        type=bool,
        default=TEST,
        help="Activate the train mode. If not, the test mode is activated"
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=100,
        help="Epoch!"
    )
    
    print("Using {} dataset....".format(DATASET))

    args = parser.parse_args()
    args.dataset_name = DATASET
    
    ## if you are using 10 Hz data and want to divide the frame to 2Hz, you have to 10/2 = 5// only int is considered, here.
    args.frame_interval = 1# skip rate
    args.train_val_test_ratio = (0.7, 0.2, 0.1)
    args.data_file = os.path.join("../data/", "{}-dataset-seed{}.cpkl".format(DATASET, args.seed))#pickle contains 'train', 'val', 'test' file name and 'scale param'
    # args.data_file = os.path.join("./data/", "dataset-seed{}.not.gaussian.cpkl".format(args.seed))
    args.load_name = "none"
    if not os.path.exists(args.data_file):
        PreprocessData(args)

    args.save_name = "{}-{}-{}.S2S_GRUmodel.seed{}.tar".format(DATASET, args.obs_length, args.pred_length, args.seed)
    args.load_name = "21.Apol-6-10.S2S_GRUmodel.seed42.tar".format(args.seed)
    
    model = VanillaGRU(args)
    # for i in range(12300, 12500):
    #     hist, hist_mask, fut, fut_mask, ref_pose, AgentInfo = model.test_dataset[i]
    #     kalman_model = KalmanModel(args)
    #     pred, _ = kalman_model.PredictTraj(hist)
    #     print("pred: ", pred)
    #     if AgentInfo[-1] == 3:
    #         plt.plot(hist[:,0], hist[:,1], "ko-", alpha=0.8)
    #         plt.plot(fut[:,0], fut[:,1], "g^-")
    #         plt.plot(pred[:len(fut),0], pred[:len(fut),2], "r^-")
    #         plt.axis('equal')

    #         plt.show()

    if args.train:
        model.train()
    
    if args.test:
        model.load()
        model.test()

    
if __name__ == "__main__":
    main()

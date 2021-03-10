import os
import math
import pickle
import random
import numpy as np
import glob
import torch
import cv2
import sys
sys.path.append("..")
import utils.util as util

#we first downsample the original images with scaling factors 0.6, 0.7, 0.8, 0.9 to generate the HR/LR images.
for scale in [0.9, 0.8, 0.7, 0.6]:
    GT_folder = '/data0/xtkong/data/DIV2K_train_HR'
    LR_folder = '/data0/xtkong/data/DIV2K_train_LR_bicubic/X4'
    save_GT_folder = '/data0/xtkong/data/DIV2K800_scale_GT'
    save_LR_folder = '/data0/xtkong/data/DIV2K800_scale_LR'
    for i in [save_GT_folder, save_LR_folder]:
        if os.path.exists(i):
            pass
        else:
            os.makedirs(i)
    img_GT_list = util._get_paths_from_images(GT_folder)
    img_LR_list = util._get_paths_from_images(LR_folder)
    assert len(img_GT_list) == len(img_LR_list), 'different length of GT_folder and LR_folder.'
    for path_GT, path_LR in zip(img_GT_list, img_LR_list):
        img_GT = cv2.imread(path_GT)
        img_LR = cv2.imread(path_LR)
        img_GT = img_GT * 1.0 / 255
        img_LR = img_LR * 1.0 / 255
        img_GT = torch.from_numpy(np.transpose(img_GT[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = torch.from_numpy(np.transpose(img_LR[:, :, [2, 1, 0]], (2, 0, 1))).float()
        # imresize

        rlt_GT = imresize(img_GT, scale, antialiasing=True)
        rlt_LR = imresize(img_LR, scale, antialiasing=True)
        print(str(scale) + "_" + os.path.basename(path_GT))
        import torchvision.utils

        torchvision.utils.save_image((rlt_GT * 255).round() / 255,
                                     os.path.join(save_GT_folder, str(scale) + "_" + os.path.basename(path_GT)), nrow=1,
                                     padding=0,
                                     normalize=False)
        torchvision.utils.save_image((rlt_LR * 255).round() / 255,
                                     os.path.join(save_LR_folder, str(scale) + "_" + os.path.basename(path_LR)), nrow=1,
                                     padding=0,
                                     normalize=False)

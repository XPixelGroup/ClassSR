import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch
from models.archs.RCAN_arch import RCAN
import numpy as np
import time

class classSR_3class_rcan(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(classSR_3class_rcan, self).__init__()
        self.upscale=4
        self.classifier=Classifier()

        self.net1 = RCAN(n_resgroups=10, n_resblocks=20, n_feats=36, res_scale=1, n_colors=3, rgb_range=1,
                         scale=4, reduction=16)
        self.net2 = RCAN(n_resgroups=10, n_resblocks=20, n_feats=50, res_scale=1, n_colors=3, rgb_range=1,
                         scale=4, reduction=16)
        self.net3 = RCAN(n_resgroups=10, n_resblocks=20, n_feats=64, res_scale=1, n_colors=3, rgb_range=1,
                         scale=4, reduction=16)

    def forward(self, x,is_train):
        if is_train:
            self.net1.eval()
            self.net2.eval()
            self.net3.eval()
            class_type = self.classifier(x/255.)
            p = F.softmax(class_type, dim=1)
            out1 = self.net1(x)
            out2 = self.net2(x)
            out3 = self.net3(x)

            p1 = p[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            p2 = p[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            p3 = p[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            out = p1 * out1 + p2 * out2 + p3 * out3
            return out, p
        else:
            for i in range(len(x)):
                type = self.classifier(x[i].unsqueeze(0)/255.) #rcan

                flag = torch.max(type, 1)[1].data.squeeze()
                p = F.softmax(type, dim=1)
                # flag=np.random.randint(0,2)
                #flag=0
                if flag == 0:
                    out = self.net1(x[i].unsqueeze(0))
                elif flag == 1:
                    out = self.net2(x[i].unsqueeze(0))
                elif flag == 2:
                    out = self.net3(x[i].unsqueeze(0))
                if i == 0:
                    out_res = out
                    type_res = p
                else:
                    out_res = torch.cat((out_res, out), 0)
                    type_res = torch.cat((type_res, p), 0)

            return out_res, type_res

        return out_res,type_res

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lastOut = nn.Linear(32, 3)

        # Condtion network
        self.CondNet = nn.Sequential(nn.Conv2d(3, 128, 4, 4), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 32, 1))
        arch_util.initialize_weights([self.CondNet], 0.1)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AvgPool2d(out.size()[2])(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        return out

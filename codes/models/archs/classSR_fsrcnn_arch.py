import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch
from models.archs.FSRCNN_arch import FSRCNN_net
import numpy as np
import time

class classSR_3class_fsrcnn_net(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(classSR_3class_fsrcnn_net, self).__init__()
        self.upscale=4
        self.classifier=Classifier()
        self.net1 = FSRCNN_net(in_nc,self.upscale,16,12,4)
        self.net2 = FSRCNN_net(in_nc,self.upscale,36,12,4)
        self.net3 = FSRCNN_net(in_nc,self.upscale,56,12,4)

    def forward(self, x,is_train):
        if is_train:
            for i in range(len(x)):
                # print(x[i].unsqueeze(0).shape)
                type = self.classifier(x[i].unsqueeze(0))
                p = F.softmax(type, dim=1)
                p1 = p[0][0]
                p2 = p[0][1]
                p3 = p[0][2]

                out1 = self.net1(x[i].unsqueeze(0))
                out2 = self.net2(x[i].unsqueeze(0))
                out3 = self.net3(x[i].unsqueeze(0))


                out = out1 * p1 + out2 * p2 + out3 * p3

                if i == 0:
                    out_res = out
                    type_res = p
                else:
                    out_res = torch.cat((out_res, out), 0)
                    type_res = torch.cat((type_res, p), 0)
        else:

            for i in range(len(x)):
                type = self.classifier(x[i].unsqueeze(0))

                flag = torch.max(type, 1)[1].data.squeeze()
                p = F.softmax(type, dim=1)
                #flag=np.random.randint(0,2)
                #flag=2
                if flag == 0:
                    out = self.net1(x[i].unsqueeze(0))
                elif flag==1:
                    out = self.net2(x[i].unsqueeze(0))
                elif flag==2:
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



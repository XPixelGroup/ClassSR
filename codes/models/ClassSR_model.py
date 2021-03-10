import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss,class_loss_3class,average_loss_3class
from torchsummary import summary
from models.archs import arch_util
import cv2
import numpy as np
from utils import util
from data import util as ut
import os.path as osp
import os

logger = logging.getLogger('base')


class ClassSR_Model(BaseModel):
    def __init__(self, opt):
        super(ClassSR_Model, self).__init__(opt)

        self.patch_size = int(opt["patch_size"])
        self.step = int(opt["step"])
        self.scale = int(opt["scale"])
        self.name = opt['name']
        self.which_model = opt['network_G']['which_model_G']


        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.l1w = float(opt["train"]["l1w"])
            self.class_loss_w = float(opt["train"]["class_loss_w"])
            self.average_loss_w = float(opt["train"]["average_loss_w"])
            self.pf = opt['logger']['print_freq']
            self.batch_size = int(opt['datasets']['train']['batch_size'])
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'ClassSR_loss':
                self.cri_pix = nn.L1Loss().to(self.device)
                self.class_loss = class_loss_3class().to(self.device)
                self.average_loss = average_loss_3class().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))


            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            if opt['fix_SR_module']:
                for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                    if v.requires_grad and "class" not in k:
                        v.requires_grad=False

            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)
        self.LQ_path = data['LQ_path'][0]
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT
            self.GT_path = data['GT_path'][0]


    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H, self.type = self.netG(self.var_L, self.is_train)
        #print(self.type)
        l_pix = self.cri_pix(self.fake_H, self.real_H)
        class_loss=self.class_loss(self.type)
        average_loss=self.average_loss(self.type)
        loss = self.l1w * l_pix + self.class_loss_w * class_loss+self.average_loss_w*average_loss

        if step % self.pf == 0:
           self.print_res(self.type)

        loss.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['class_loss'] = class_loss.item()
        self.log_dict['average_loss'] = average_loss.item()
        self.log_dict['loss'] = loss.item()

    def test(self):
        self.netG.eval()
        self.var_L = cv2.imread(self.LQ_path, cv2.IMREAD_UNCHANGED)
        self.real_H = cv2.imread(self.GT_path, cv2.IMREAD_UNCHANGED)

        lr_list, num_h, num_w, h, w = self.crop_cpu(self.var_L, self.patch_size, self.step)
        gt_list=self.crop_cpu(self.real_H,self.patch_size*4,self.step*4)[0]
        sr_list = []
        index = 0

        psnr_type1 = 0
        psnr_type2 = 0
        psnr_type3 = 0

        for LR_img,GT_img in zip(lr_list,gt_list):

            if self.which_model=='classSR_3class_rcan':
                img = LR_img.astype(np.float32)
            else:
                img = LR_img.astype(np.float32) / 255.
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            # some images have 4 channels
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img = img[:, :, [2, 1, 0]]
            img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()[None, ...].to(
                self.device)
            srt, type = self.netG(img, False)

            if self.which_model == 'classSR_3class_rcan':
                sr_img = util.tensor2img(srt.detach()[0].float().cpu(), out_type=np.uint8, min_max=(0, 255))
            else:
                sr_img = util.tensor2img(srt.detach()[0].float().cpu())
            sr_list.append(sr_img)

            if index == 0:
                type_res = type
            else:
                type_res = torch.cat((type_res, type), 0)

            psnr=util.calculate_psnr(sr_img, GT_img)
            flag=torch.max(type, 1)[1].data.squeeze()
            if flag == 0:
                psnr_type1 += psnr
            if flag == 1:
                psnr_type2 += psnr
            if flag == 2:
                psnr_type3 += psnr

            index += 1

        self.fake_H = self.combine(sr_list, num_h, num_w, h, w, self.patch_size, self.step)
        self.real_H = self.real_H[0:h * self.scale, 0:w * self.scale, :]
        self.num_res = self.print_res(type_res)
        self.psnr_res=[psnr_type1,psnr_type2,psnr_type3]


        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L
        out_dict['rlt'] = self.fake_H
        out_dict['num_res'] = self.num_res
        out_dict['psnr_res']=self.psnr_res
        if need_GT:
            out_dict['GT'] = self.real_H
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        load_path_classifier = self.opt['path']['pretrain_model_classifier']
        load_path_G_branch3 = self.opt['path']['pretrain_model_G_branch3']
        load_path_G_branch2= self.opt['path']['pretrain_model_G_branch2']
        load_path_G_branch1 = self.opt['path']['pretrain_model_G_branch1']
        load_path_Gs=[load_path_G_branch1,load_path_G_branch2,load_path_G_branch3]
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        if load_path_classifier is not None:
            logger.info('Loading model for classfier [{:s}] ...'.format(load_path_classifier))
            self.load_network_classifier_rcan(load_path_classifier, self.netG, self.opt['path']['strict_load'])
        if load_path_G_branch3 is not None and load_path_G_branch1 is not None and load_path_G_branch2 is not None:
            logger.info('Loading model for branch1 [{:s}] ...'.format(load_path_G_branch1))
            logger.info('Loading model for branch2 [{:s}] ...'.format(load_path_G_branch2))
            logger.info('Loading model for branch3 [{:s}] ...'.format(load_path_G_branch3))
            self.load_network_classSR_3class(load_path_Gs, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def crop_cpu(self,img,crop_sz,step):
        n_channels = len(img.shape)
        if n_channels == 2:
            h, w = img.shape
        elif n_channels == 3:
            h, w, c = img.shape
        else:
            raise ValueError('Wrong image shape - {}'.format(n_channels))
        h_space = np.arange(0, h - crop_sz + 1, step)
        w_space = np.arange(0, w - crop_sz + 1, step)
        index = 0
        num_h = 0
        lr_list=[]
        for x in h_space:
            num_h += 1
            num_w = 0
            for y in w_space:
                num_w += 1
                index += 1
                if n_channels == 2:
                    crop_img = img[x:x + crop_sz, y:y + crop_sz]
                else:
                    crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
                lr_list.append(crop_img)
        h=x + crop_sz
        w=y + crop_sz
        return lr_list,num_h, num_w,h,w

    def combine(self,sr_list,num_h, num_w,h,w,patch_size,step):
        index=0
        sr_img = np.zeros((h*self.scale, w*self.scale, 3), 'float32')
        for i in range(num_h):
            for j in range(num_w):
                sr_img[i*step*self.scale:i*step*self.scale+patch_size*self.scale,j*step*self.scale:j*step*self.scale+patch_size*self.scale,:]+=sr_list[index]
                index+=1
        sr_img=sr_img.astype('float32')

        for j in range(1,num_w):
            sr_img[:,j*step*self.scale:j*step*self.scale+(patch_size-step)*self.scale,:]/=2

        for i in range(1,num_h):
            sr_img[i*step*self.scale:i*step*self.scale+(patch_size-step)*self.scale,:,:]/=2
        return sr_img

    def print_res(self, type_res):
        num0 = 0
        num1 = 0
        num2 = 0

        for i in torch.max(type_res, 1)[1].data.squeeze():
            if i == 0:
                num0 += 1
            if i == 1:
                num1 += 1
            if i == 2:
                num2 += 1

        return [num0, num1,num2]



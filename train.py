import os
import yaml
import shutil
import pickle

from tqdm import tqdm

import torch
import torch.utils.data

from model import PoseDepth

class Trainer(object):
    def __init__(self, config, train_loader=None):
        self.config = config
        self.dataset = config.dataset
        self.ckpt_root = config.ckpt_root
        self.ckpt_dir = config.ckpt_dir
        self.ckpt_name = '{}'.format(config.seed)
        self.train_loader = train_loader

        torch.cuda.set_device(self.config.gpu)
        # load model and optimizer
        self.model = PoseDepth(self.config)
        self.model = self.model.cuda()

        print('Number of model parameters: {:,}'.format(sum([p.data.nelement() for p in self.model.depth_encoder.parameters()])))	
        print('Number of model parameters: {:,}'.format(sum([p.data.nelement() for p in self.model.depth_decoder.parameters()])))

        
        self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.parameters()),
                                            'lr': self.config.init_lr}])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                 milestones=config.milestones,
                                                                 gamma=config.lr_factor)

        if int(self.config.start_epoch) > 0:
            self.config.start_epoch, \
            self.model, \
            self.optimizer, \
            self.lr_scheduler = self.load_checkpoint(int(self.config.start_epoch),
                                                     self.model,
                                                     self.optimizer,
                                                     self.lr_scheduler)

    def train(self):
        print("\nTrain on {} samples".format(len(self.train_loader)))
        self.save_checkpoint(0,
                             self.model,
                             self.optimizer,
                             self.lr_scheduler)
        for epoch in range(self.config.start_epoch, self.config.max_epoch):
            print("\nEpoch: {}/{}".format(epoch+1, self.config.max_epoch))
            # train for one epoch
            self.train_one_epoch(epoch)
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.save_checkpoint(epoch+1,
                                 self.model,
                                 self.optimizer,
                                 self.lr_scheduler)
        
    def train_one_epoch(self, epoch):
        self.model.train()
        for (i, inputs) in enumerate(tqdm(self.train_loader)):
            for k, v in inputs.items():
                inputs[k] = v.cuda()

            losses = self.model.forward(inputs)
            loss = self.config.reproj_weight * losses['reproj_loss'] + \
                   self.config.smooth_weight * losses['smooth_loss']
        
            # training information
            msg_batch = "Epoch:{} Iter:{} " \
                        "reproj_loss={:.4f}," \
                        "smooth_loss={:.4f}, " \
                        "loss={:.4f} " \
                        .format((epoch + 1), i,\
                                 losses['reproj_loss'].data,
                                 losses['smooth_loss'].data,
                                 loss.data)   

            if (i % self.config.display) == 0:
                print(msg_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_checkpoint(self, epoch, model, optimizer, lr_scheduler):
        filename = self.ckpt_name + '_' + str(epoch) + '.pth'
        save_path = os.path.join(self.ckpt_root, self.dataset, self.ckpt_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(
            {'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()},
            os.path.join(save_path, filename))

    def load_checkpoint(self, epoch, model, optimizer, lr_scheduler):
        filename = self.ckpt_name + '_' + str(epoch) + '.pth'
        ckpt = torch.load(os.path.join(self.ckpt_root, self.dataset, self.ckpt_dir, filename))
        epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

        print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt['epoch']))

        return epoch, model, optimizer, lr_scheduler
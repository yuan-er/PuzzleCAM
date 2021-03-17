# -*- coding: utf-8 -*-
# created by wq(wuqiang@rockontrol.com)
""" Main entry point to the training process of PuzzleCAM with Pytorch Lightning"""
import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.networks import Classifier
from core.puzzle_utils import tile_features, merge_features
from tools.ai.torch_utils import L1_Loss, L2_Loss, shannon_entropy_loss, make_cam
from tools.ai.optim_utils import PolyOptimizer
from tools.general.io_utils import str2bool


class Trainer(pl.LightningModule):
    def __init__(self, arguments, train_dataset, val_dataset=None):
        print("Initializing %s: (args: %s", self.__class__.__name__, self._args)
        super(Trainer, self).__init__()
        self._args = arguments

        # data loader
        self.train_loader = DataLoader(train_dataset, batch_size=self._args.batch_size,
                                       num_workers=self._args.num_workers, shuffle=True,
                                       drop_last=True)
        val_iteration = len(self.train_loader)
        self.max_iteration = self._args.max_epoch * val_iteration

        # if val_dataset is not None:
        #     train_dataset_for_seg = val_dataset
        #     self.train_loader_for_seg = DataLoader(train_dataset_for_seg, batch_size=self._args.batch_size,
        #                                            num_workers=1, drop_last=True)

        # network
        self.model = Classifier(self._args.architecture, self._args.num_classes, mode=self._args.mode)
        self.gap_fn = self.model.global_average_pooling_2d

        # loss
        self.class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()
        if self._args.re_loss == 'L1_Loss':
            self.re_loss_fn = L1_Loss
        else:
            self.re_loss_fn = L2_Loss

        self.loss_option = self._args.loss_option.split('_')

        print("Initialized %s", self.__class__.__name__)

    def train_dataloader(self):
        return self.train_loader

    # def val_dataloader(self):
    #     return self.train_loader_for_seg

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()

        # Normal classification
        logits, features = self.model(images, with_cam=True)

        # Puzzle Module
        tiled_images = tile_features(images, self._args.num_pieces)
        tiled_logits, tiled_features = self.model(tiled_images, with_cam=True)
        re_features = merge_features(tiled_features, self._args.num_pieces, self._args.batch_size)

        # Losses
        class_loss = self.class_loss_fn(logits, labels).mean()

        if 'pcl' in self.loss_option:
            p_class_loss = self.class_loss_fn(self.gap_fn(re_features), labels).mean()
        else:
            p_class_loss = torch.zeros(1).cuda()

        if 're' in self.loss_option:
            if self._args.re_loss_option == 'masking':
                class_mask = labels.unsqueeze(2).unsqueeze(3)
                re_loss = self.re_loss_fn(features, re_features) * class_mask
                re_loss = re_loss.mean()
            elif self._args.re_loss_option == 'selection':
                re_loss = 0.
                for b_index in range(labels.size()[0]):
                    class_indices = labels[b_index].nonzero(as_tuple=True)
                    selected_features = features[b_index][class_indices]
                    selected_re_features = re_features[b_index][class_indices]

                    re_loss_per_feature = self.re_loss_fn(selected_features, selected_re_features).mean()
                    re_loss += re_loss_per_feature
                re_loss /= labels.size()[0]
            else:
                re_loss = self.re_loss_fn(features, re_features).mean()
        else:
            re_loss = torch.zeros(1).cuda()

        if 'conf' in self.loss_option:
            conf_loss = shannon_entropy_loss(tiled_logits)
        else:
            conf_loss = torch.zeros(1).cuda()

        if self._args.alpha_schedule == 0.0:
            alpha = self._args.alpha
        else:
            alpha = min(self._args.alpha * batch_idx / (self.max_iteration * self._args.alpha_schedule),
                        self._args.alpha)

        loss = class_loss + p_class_loss + alpha * re_loss + conf_loss
        tensorboard_logs = {'loss': loss.item(),
                            'class_loss': class_loss.item(),
                            'p_class_loss': p_class_loss.item(),
                            're_loss': re_loss.item(),
                            'conf_loss': conf_loss.item(),
                            'alpha': alpha,
                            }

        return {'loss': loss, 'log': tensorboard_logs}

    # def validation_step(self, batch, batch_idx):
    #     images, labels, gt_masks = batch
    #     images = images.cuda()
    #     labels = labels.cuda()
    #
    #     _, features = self.model(images, with_cam=True)
    #
    #     # features = resize_for_tensors(features, images.size()[-2:])
    #     # gt_masks = resize_for_tensors(gt_masks, features.size()[-2:], mode='nearest')
    #
    #     mask = labels.unsqueeze(2).unsqueeze(3)
    #     cams = (make_cam(features) * mask)
    #     loss = None
    #     return {'val_loss': loss}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #
    #     tensorboard_logs = {'avg_val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        param_groups = self.model.get_parameter_groups(print_fn=None)

        optimizer = PolyOptimizer([
            {'params': param_groups[0], 'lr': self._args.lr, 'weight_decay': self._args.wd},
            {'params': param_groups[1], 'lr': 2 * self._args.lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': 10 * self._args.lr, 'weight_decay': self._args.wd},
            {'params': param_groups[3], 'lr': 20 * self._args.lr, 'weight_decay': 0},
        ], lr=self._args.lr, momentum=0.9, weight_decay=self._args.wd, max_step=self.max_iteration,
            nesterov=self._args.nesterov)

        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset config
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

    # Network config
    parser.add_argument('--architecture', default='resnet50', type=str)
    parser.add_argument('--mode', default='normal', type=str)  # fix

    # Hyperparameter config
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=15, type=int)

    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=True, type=str2bool)

    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--min_image_size', default=320, type=int)
    parser.add_argument('--max_image_size', default=640, type=int)

    parser.add_argument('--augment', default='', type=str)

    # For Puzzle-CAM
    parser.add_argument('--num_pieces', default=4, type=int)

    # 'cl_pcl', 'cl_re', 'cl_conf', 'cl_pcl_re', 'cl_pcl_re_conf'
    parser.add_argument('--loss_option', default='cl_pcl_re', type=str)

    parser.add_argument('--level', default='feature', type=str)

    parser.add_argument('--re_loss', default='L1_Loss', type=str)  # 'L1_Loss', 'L2_Loss'
    parser.add_argument('--re_loss_option', default='masking', type=str)  # 'none', 'masking', 'selection'

    parser.add_argument('--alpha', default=4.0, type=float)
    parser.add_argument('--alpha_schedule', default=0.50, type=float)

    args = parser.parse_args()
    train_app = Trainer(args)
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(monitor='train_loss',
                                          filepath='{epoch:02d}-{loss:.4f}',
                                          save_top_k=3,
                                          verbose=True)

    trainer = pl.Trainer(auto_lr_find=True,
                         gpus=args.gpus,
                         distributed_backend='dp',
                         checkpoint_callback=checkpoint_callback)

    trainer.fit(train_app)

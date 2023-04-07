import os
import torch
from torch import nn
from third_party import networks
from models import text_models
import models.operator as op
from models.localizer import LocalizerAttn
import functools
import torchvision
from tensorboardX import SummaryWriter


def pairwise_dis(x, y):
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)
    return torch.abs(x - y).sum(dim=1)


class NeuralOperator(nn.Module):
    def __init__(self, opts):
        super(NeuralOperator, self).__init__()
        self.isTrain = self.training
        self.img_dim = opts.img_dim

        self.n_downsampling = opts.n_downsampling
        if self.isTrain:
            self.w_gate = opts.w_gate

        self.img_E = networks.ContentEncoder(n_downsample=self.n_downsampling,
                                             n_res=0,
                                             input_dim=self.img_dim,
                                             dim=64,
                                             norm='in', activ='relu', pad_type='reflect')

        self.text_E = text_models.BertTextEncoder(pretrained=True, img_dim=self.img_E.output_dim)

        self.localizer = LocalizerAttn(img_dim=self.img_E.output_dim, text_dim=512)

        # operator
        if opts.operator == 'adaroute':
            self.operator = op.Adaptive_Routing(n_res=opts.num_adablock, dim=self.img_E.output_dim, text_dim=512,
                                                temperature=opts.temperature)
        else:
            raise Exception('no such operator %s' % (opts.operator))

        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            self.params = list(self.localizer.parameters()) + list(self.operator.parameters()) + list(
                self.img_E.parameters())
            self.lr = opts.lr
            # self.opt = torch.optim.Adam([{'params': self.text_E.parameters(), 'lr': opts.lr / 10.},
            #                              {'params': params}], lr=opts.lr, betas=(0.5, 0.999))

    def set_input(self, src_img, text, tgt_img):
        self.real_A = src_img
        self.text = text
        self.real_B = tgt_img

    def forward(self, use_gt_attn_rate=0., temperature_rate=0.):
        batch_size = self.real_A.size(0)

        # image feature
        feat = self.img_E(torch.cat([self.real_A, self.real_B], dim=0))
        real_A_feat, real_B_feat = torch.split(feat, batch_size, dim=0)
        self.real_B_feat = real_B_feat.detach()

        # text feature
        img1d = torch.mean(real_A_feat, dim=(2, 3))
        self.text1, self.text2, self.text_tokens, rawtext = self.text_E.extract_text_feature(self.text, img1d)

        # ground-truth attention mask
        with torch.no_grad():
            diff = torch.mean(torch.abs(real_A_feat - real_B_feat), dim=1).view(batch_size, -1)
            diff = diff - torch.min(diff, dim=1, keepdim=True)[0].expand_as(diff)
            self.attn_gt = (diff / (torch.max(diff, dim=1, keepdim=True)[0] + 1e-5)).view(batch_size, 1,
                                                                                          real_A_feat.size(2),
                                                                                          real_A_feat.size(3))

        # attention mask
        # self.attn = self.attn_gt # for sanity check
        self.attn = self.localizer(real_A_feat, self.text1[0])

        # schedule samping
        if self.isTrain and use_gt_attn_rate > 0:
            use_gt_attn = torch.rand(batch_size, 1, 1, 1, device=self.attn_gt.device)
            use_gt_attn = torch.lt(use_gt_attn, use_gt_attn_rate).float().expand_as(self.attn_gt)
            attn = use_gt_attn * self.attn_gt + (1 - use_gt_attn) * self.attn
        else:
            attn = self.attn

        # edit feature
        edit_feat, self.gates = self.operator(real_A_feat, self.text2[0], temperature_rate)

        # fuse original and edited feature
        self.fake_B_feat = torch.mul(edit_feat, attn) + torch.mul(real_A_feat, (1 - attn))

        return self.fake_B_feat

    def backward(self):
        # attention loss
        self.loss_G_attn = self.criterionL1(self.attn, self.attn_gt)

        # gate divergence loss for visualization purpose, set W=0 in the bash script if you don't need it
        t2 = self.text2[0].detach()
        self.loss_disgate = -((pairwise_dis(self.gates, torch.cat((self.gates[1:], self.gates[:1]), dim=0))) / (
                pairwise_dis(t2, torch.cat((t2[1:], t2[:1]), dim=0)) + 1e-5) / 3. + (
                                  pairwise_dis(self.gates, torch.cat((self.gates[2:], self.gates[:2]), dim=0))) / (
                                      pairwise_dis(t2, torch.cat((t2[2:], t2[:2]), dim=0)) + 1e-5) / 3. + (
                                  pairwise_dis(self.gates, torch.cat((self.gates[3:], self.gates[:3]), dim=0))) / (
                                      pairwise_dis(t2, torch.cat((t2[3:], t2[:3]), dim=0)) + 1e-5) / 3.)
        self.loss_disgate = self.loss_disgate.mean()

        # feat loss
        feat_dis = torch.mean(torch.abs(self.fake_B_feat - self.real_B_feat), dim=1, keepdim=True)
        weighted_feat_dis = torch.sum(feat_dis * self.attn_gt, dim=(1, 2, 3)) / (
                torch.sum(self.attn_gt, dim=(1, 2, 3)) + 1e-5)
        self.loss_G_feat = torch.mean(weighted_feat_dis)

        # summation
        self.loss_G = self.loss_G_feat * 10 + self.loss_G_attn * 100 + self.loss_disgate * self.w_gate

        return self.loss_G

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def collect_params(self):
        param1 = {'params': self.text_E.parameters(), 'lr': self.lr / 10}
        param2 = {'params': self.params, 'lr': self.lr, 'betas': (0.5, 0.999)}

        return param1, param2

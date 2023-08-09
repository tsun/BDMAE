import argparse
import logging
import os
import sys

sys.path.append('/')
sys.path.append(os.getcwd())

from mae import models_mae

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import get_transform, get_dataset_norm_stats
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.dataset.bd_dataset import prepro_cls_DatasetBD
from utils.save_load_attack import load_attack_result
import yaml
from pprint import pformat

from pytorch_ssim import SSIM

class BDMAEmodel(nn.Module):
    def __init__(self, args):
        super(BDMAEmodel, self).__init__()
        self.base_model = generate_cls_model(model_name=args.model, num_classes=args.num_classes)

        self.mae_model = getattr(models_mae, args.mae_arch)()
        checkpoint = torch.load(args.mae_ckp, map_location='cpu')
        msg = self.mae_model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        self.data_mean, self.data_std = get_dataset_norm_stats(args.dataset)
        self.data_mean = torch.tensor(self.data_mean).reshape(1,3,1,1).to(args.device)
        self.data_std = torch.tensor(self.data_std).reshape(1,3,1,1).to(args.device)
        self.in_mean = torch.tensor(np.array([0.485, 0.456, 0.406])).float().reshape(1,3,1,1).to(args.device)
        self.in_std = torch.tensor(np.array([0.229, 0.224, 0.225])).float().reshape(1,3,1,1).to(args.device)

        self.in_sz = (224, 224)
        self.token_sz = (14, 14)
        self.data_sz = (args.input_height,args.input_width)
        self.mask_ratio = args.mask_ratio
        self.alpha = args.alpha
        self.rp_init_in = args.rp_init_in
        self.rp_init_out = args.rp_init_out
        self.rp_refine = args.rp_refine
        self.topo_po = args.topo_po

    def load_base_model(self, result):
        self.base_model.load_state_dict(result['model'])

    def normalize(self, x, mean, std):
        return (x-mean.to(x.device)) / std.to(x.device)

    def unnormalize(self, x, mean, std):
        return (x * std.to(x.device) + mean.to(x.device))

    def data_normalize(self, x):
        return self.normalize(x, self.data_mean, self.data_std)

    def data_unnormalize(self, x):
        return self.unnormalize(x, self.data_mean, self.data_std)

    def in_normalize(self, x):
        return self.normalize(x, self.in_mean, self.in_std)

    def in_unnormalize(self, x):
        return self.unnormalize(x, self.in_mean, self.in_std)

    def forward(self, x):
        outputs = self.base_model(x)
        return outputs

    def cpredict(self, x, xn, cmask):
        with torch.no_grad():
            loss, y, mask = self.mae_model(xn, mask_ratio=self.mask_ratio, mask_type='precom', mask=cmask)
            y = self.mae_model.unpatchify(y)

            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, self.mae_model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
            mask = self.mae_model.unpatchify(mask)  # 1 is removing, 0 is keeping

        mask = F.interpolate(mask, size=self.data_sz, mode='nearest')
        y = F.interpolate(y, size=self.data_sz, mode='nearest')
        y = self.in_unnormalize(y)
        xn_mae = x * (1 - mask) + y * mask

        xn_mae = self.data_normalize(xn_mae)
        outputs = self.base_model(xn_mae)
        return outputs, xn_mae, mask

    def gen_local_sample(self, S, ids_rem, len, Adj):
        # randomly generate seed token
        N, H, W = S.shape
        S = S.reshape((N, H * W))
        rnd = np.random.random((N, H * W)).astype(np.float32) * (S)
        ids_seed = np.argmax(rnd, axis=1)

        # iteratively collect neighbors
        invalid = np.zeros((N, H * W)).astype(np.float32)
        np.put_along_axis(invalid, ids_rem, 1, axis=1)
        selected = np.zeros((N, H * W)).astype(np.float32)
        candidate = np.zeros((N, H * W)).astype(np.float32)
        np.put_along_axis(selected, ids_seed.reshape(N, 1), 1, axis=1)
        candidate += Adj[ids_seed]
        for _ in range(len - 1):
            rnd = np.random.random((N, H * W)).astype(np.float32) * (
                        S + candidate * self.topo_po) - invalid * 100 - selected * 100
            ids_sel = np.argmax(rnd, axis=1)
            np.put_along_axis(selected, ids_sel.reshape(N, 1), 1, axis=1)
            candidate += Adj[ids_sel]
            candidate = np.clip(candidate, 0, 1)
            candidate -= selected

        return selected


    def forward_mae(self, x):
        # predict on original x
        outputs_org = self.base_model(x)
        probs_org = F.softmax(outputs_org, dim=-1)
        tgt_lbls = probs_org.argmax(-1)

        # unnormalize
        x = self.data_unnormalize(x)

        # resize to in_sz
        xn = F.interpolate(x, size=self.in_sz, mode='bilinear')

        # in normalize
        xn = self.in_normalize(xn)

        N = x.shape[0]
        H = W = 14

        p = 0.75
        all_xn_mae = []
        all_mask = []
        all_outputs = []
        rp_init_in = self.rp_init_in
        rp_init_out = self.rp_init_out
        len_mask = int(H * W * p)

        all_I = []
        diff_pred_cnt = np.zeros((N, H, W))

        for kk in range(rp_init_out):
            rnd = [np.random.random((1, H * W)).astype(np.float32) for _ in range(rp_init_in)]
            ids_sort = [np.argsort(-rd, axis=1) for rd in rnd]
            ids_sort = np.concatenate(ids_sort, axis=1)

            for ii in range(rp_init_in):
                ids_mask = ids_sort[:, ii*len_mask:(ii+1)*len_mask]
                mask = np.zeros((1, H * W), np.float32)
                mask[:, ids_mask] = 1
                mask = np.repeat(mask, N, axis=0)

                with torch.no_grad():
                    outputs, xn_mae, mask_mae = self.cpredict(x, xn, mask)
                    all_xn_mae.append(xn_mae.detach().cpu())
                    all_outputs.append(outputs.detach().cpu())
                    all_mask.append(mask_mae.detach().cpu())

                beta = ((outputs.argmax(-1) != tgt_lbls).float().cpu().numpy()).reshape(N, 1, 1)
                diff_pred_cnt += mask.reshape(N, H, W) * beta

            xn_mae_agg = torch.zeros(x.shape)
            cnt_egg = torch.zeros(x.shape)
            for mask_mae, xn_mae in zip(all_mask, all_xn_mae):
                xn_mae_agg += (mask_mae * xn_mae)
                cnt_egg += mask_mae

            xn_mae_agg = xn_mae_agg / cnt_egg
            I = SSIM()(x.cpu(), self.data_unnormalize(xn_mae_agg))
            all_I.append(I)


        scores_i = [1 - F.interpolate(I.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1).numpy() for I in all_I]
        scores_i = np.stack(scores_i).mean(0)
        scores_l = diff_pred_cnt / (rp_init_in * rp_init_out)

        # build neighborhood graph
        Adj = np.zeros((H * W, H * W), dtype=bool)
        for hh in range(H):
            for ww in range(W):
                cen = hh*(W)+ww
                if hh>0:
                    nei = (hh-1)*(W)+ww
                    Adj[cen, nei] = True
                if hh<H-1:
                    nei = (hh+1)*(W)+ww
                    Adj[cen, nei] = True
                if ww>0:
                    nei = (hh)*(W)+ww-1
                    Adj[cen, nei] = True
                if ww<W-1:
                    nei = (hh)*(W)+ww+1
                    Adj[cen, nei] = True

        cur_len_mask = int((scores_i >= 0.2).mean(0).sum())
        for ii in range(self.rp_refine):
            rnd = (scores_i)
            rnd = rnd.reshape(N, H * W)
            ids_sort = np.argsort(-rnd, axis=1)
            ids_mask = ids_sort[:, :cur_len_mask]
            ids_rem = ids_sort[:, cur_len_mask:]

            mask_full = np.zeros((N, H * W), np.float32)
            np.put_along_axis(mask_full, ids_mask, 1, axis=1)
            mask_full = np.reshape(mask_full, (N, H, W))

            mask = self.gen_local_sample(scores_i, ids_rem, cur_len_mask//2, Adj)
            mask = np.reshape(mask, (N, H, W))
            mask_rem = mask_full - mask

            with torch.no_grad():
                outputs, xn_mae, mask_mae = self.cpredict(x, xn, mask)

            beta = (2*(outputs.argmax(-1)==tgt_lbls).float().cpu().numpy()-1).reshape(N, 1, 1) * 0.05

            delta_scores = - beta*(mask) + beta*mask_rem
            scores_i = scores_i + delta_scores

        cur_len_mask = int(scores_l.mean() * H * W)
        for ii in range(self.rp_refine):
            rnd = (scores_l)
            rnd = rnd.reshape(N, H * W)
            ids_sort = np.argsort(-rnd, axis=1)
            ids_mask = ids_sort[:, :cur_len_mask]
            ids_rem = ids_sort[:, cur_len_mask:]

            mask_full = np.zeros((N, H * W), np.float32)
            np.put_along_axis(mask_full, ids_mask, 1, axis=1)
            mask_full = np.reshape(mask_full, (N, H, W))

            mask = self.gen_local_sample(scores_l, ids_rem, cur_len_mask // 2, Adj)
            mask = np.reshape(mask, (N, H, W))
            mask_rem = mask_full - mask

            with torch.no_grad():
                outputs, xn_mae, mask_mae = self.cpredict(x, xn, mask)

            beta = (2 * (outputs.argmax(-1) == tgt_lbls).float().cpu().numpy() - 1).reshape(N, 1, 1) * 0.05
            delta_scores = - beta * (mask) + beta * mask_rem
            scores_l = scores_l + delta_scores

        scores = (self.alpha * scores_i + (1-self.alpha) * scores_l)

        all_xn_mae = []
        all_mask = []
        init = 0.6
        rp = 5
        delta = 0.05
        thrs = 1 - init + np.arange(rp) * delta
        cnt=0
        while (scores >= (1 - thrs[-1])).mean() >= 0.25 and cnt < 200:
            thrs = thrs * 0.99
            cnt += 1

        for ii in range(rp):
            cur_len_mask = int((scores >= 1 - thrs[ii]).mean(0).sum())
            rnd = (scores)
            rnd = rnd.reshape(N, H * W)
            ids_sort = np.argsort(-rnd, axis=1)
            ids_mask = ids_sort[:, :cur_len_mask]
            mask = np.zeros((N, H * W), np.float32)
            np.put_along_axis(mask, ids_mask, 1, axis=1)
            mask = mask.reshape(N, H, W)

            with torch.no_grad():
                outputs, xn_mae, mask_mae = self.cpredict(x, xn, mask)

            all_xn_mae.append(xn_mae.detach().cpu())
            all_outputs.append(outputs.detach().cpu())
            all_mask.append(mask_mae.detach().cpu())

        xn_mae_agg = torch.zeros(x.shape)
        cnt_egg = torch.zeros(x.shape)
        for mask_mae, xn_mae in zip(all_mask, all_xn_mae):
            xn_mae_agg += (mask_mae * xn_mae)
            cnt_egg += mask_mae

        cnt_egg = torch.clip(cnt_egg, min=1)
        xn_mae_agg = xn_mae_agg / cnt_egg
        xn_mae_agg = xn_mae_agg * all_mask[-1] + (all_xn_mae[-1] * (1 - all_mask[-1]))

        with torch.no_grad():
            outputs = self.base_model(xn_mae_agg.cuda())

        return outputs





def get_args():
    #set the basic parameter
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str)
    parser.add_argument('--checkpoint_save', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument("--data_root", type=str)

    parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny') 
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    parser.add_argument("--input_channel", type=int)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--num_workers", type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')

    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)

    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--seed', type=str, help='random seed')
    parser.add_argument('--index', type=str, help='index of clean data')
    parser.add_argument('--result_file', type=str, help='the location of result')

    parser.add_argument('--yaml_path', type=str, default="./config/defense/ft/config.yaml", help='the path of yaml')

    parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')

    parser.add_argument('--mae_arch', type=str, default='mae_vit_base_patch16')
    parser.add_argument('--mae_ckp', type=str, default='mae_visualize_vit_base.pth')
    parser.add_argument('--mae_num', type=int, default=1)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--rp_init_in', type=int, default=5)
    parser.add_argument('--rp_init_out', type=int, default=5)
    parser.add_argument('--rp_refine', type=int, default=10)
    parser.add_argument('--topo_po', type=float, default=0.5)

    arg = parser.parse_args()

    print(arg)
    return arg


def defense(args, result):
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    if args.log_file_name is not None:
        fileHandler = logging.FileHandler(os.getcwd() + args.log + '/' + args.log_file_name.split('/')[-1] + '.log')
    else:
        if args.log is not None and args.log != '':
            fileHandler = logging.FileHandler(os.getcwd() + args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        else:
            fileHandler = logging.FileHandler(os.getcwd() + './log' + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))

    fix_random(args.seed)

    # Prepare model, optimizer, scheduler
    model = BDMAEmodel(args=args)
    model.load_base_model(result)
    model.to(args.device)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train= False, norm=True)
    x = result['bd_test']['x']
    y = result['bd_test']['y']
    data_bd_test = list(zip(x,y))
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=False,pin_memory=True)

    x = result['bd_test']['x']
    y = result['bd_test']['original_targets']
    data_bd_test_org = list(zip(x, y))
    data_bd_testset_org = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test_org,
        poison_idx=np.zeros(len(data_bd_test_org)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_bd_loader_org = torch.utils.data.DataLoader(data_bd_testset_org, batch_size=args.batch_size,
                                                 num_workers=args.num_workers, drop_last=False, shuffle=False,
                                                 pin_memory=True)



    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_clean_test = list(zip(x,y))
    data_clean_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_test,
        poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=False,pin_memory=True)


    model.eval()
    asr_acc = 0
    for i, (inputs, labels) in enumerate(data_bd_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model(inputs)
        pre_label = torch.max(outputs, dim=1)[1]
        asr_acc += torch.sum(pre_label == labels)
    original_ASR = asr_acc.item() / len(data_bd_loader.dataset) * 100

    clean_acc = 0
    for i, (inputs, labels) in enumerate(data_clean_loader):  # type: ignore
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model(inputs)
        pre_label = torch.max(outputs, dim=1)[1]
        clean_acc += torch.sum(pre_label == labels)
    original_ACC = clean_acc.item() / len(data_clean_loader.dataset) * 100

    logging.info("original ACC is {} and original ASR is {}".format(original_ACC, original_ASR))

    clean_acc = 0.
    cnt = 0.
    for i, (inputs, labels) in enumerate(data_clean_loader):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model.forward_mae(inputs)
        pre_label = torch.max(outputs.detach().cpu(), dim=1)[1]
        clean_acc += torch.sum(pre_label == labels.cpu())
        cnt += len(labels)
    mae_ACC = clean_acc.item() / len(data_clean_loader.dataset) * 100
    logging.info("ACC w/ BDMAE is {}".format(mae_ACC))

    asr_acc = 0.
    cnt = 0.
    all_preds = []
    for i, (inputs, labels) in enumerate(data_bd_loader_org):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model.forward_mae(inputs)
        pre_label = torch.max(outputs.detach().cpu(), dim=1)[1]
        asr_acc += torch.sum(pre_label == labels.cpu())
        cnt += len(labels)
        all_preds.append(pre_label.detach())
    mae_ACC_org = asr_acc.item() / len(data_bd_loader_org.dataset) * 100
    logging.info("ACC on BAD IMAGES w/ BDMAE is {}".format(mae_ACC_org))

    all_labels = []
    for i, (inputs, labels) in enumerate(data_bd_loader):
        all_labels.append(labels)
    all_labels = torch.cat(all_labels)
    mae_ASR = (torch.cat(all_preds)==all_labels).float().mean().item() * 100
    logging.info("ASR w/ BDMAE is {}".format(mae_ASR))



if __name__ == '__main__':
    ### 1. basic setting: args
    args = get_args()
    with open(args.yaml_path, 'r') as stream: 
        config = yaml.safe_load(stream) 
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    if args.dataset == "mnist":
        args.num_classes = 10
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
    elif args.dataset == "cifar10":
        args.num_classes = 10
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "cifar100":
        args.num_classes = 100
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "gtsrb":
        args.num_classes = 43
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
    elif args.dataset == "celeba":
        args.num_classes = 8
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    elif args.dataset == "tiny":
        args.num_classes = 200
        args.input_height = 64
        args.input_width = 64
        args.input_channel = 3
    elif "imagenet10" in args.dataset:
        args.num_classes = 10
        args.input_height = 224
        args.input_width = 224
        args.input_channel = 3
    elif args.dataset == 'VGGFace2':
        args.num_classes = 170
        args.input_height = 224
        args.input_width = 224
        args.input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    
    save_path = '/record/' + args.result_file
    if args.checkpoint_save is None:
        args.checkpoint_save = save_path + '/record/defence/ft/'
    if args.log is None:
        args.log = save_path + '/saved/ft/'
    else:
        args.log_file_name = args.result_file + '_' + str(args.seed)
    if not (os.path.exists(os.getcwd() + args.log)):
        os.makedirs(os.getcwd() + args.log)
    args.save_path = save_path

    ### 2. attack result(model, train data, test data)
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt', load_val=False, load_train=False)

    ### 3. defense:
    defense(args, result)

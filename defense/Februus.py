import argparse
import logging
import os
import sys

sys.path.append('/')
sys.path.append(os.getcwd())

import time
import torch
import torch.nn as nn
import numpy as np

from utils.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import get_transform, get_dataset_norm_stats
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.dataset.bd_dataset import prepro_cls_DatasetBD
from utils.save_load_attack import load_attack_result
import yaml
from pprint import pformat

from models.CompletionNetwork import CompletionNetwork
import cv2

import os
count = os.cpu_count()
torch.set_num_threads(min(count//2, 32))

def poisson_blend_old(input, output, mask):
    """
    * inputs:
        - input (torch.Tensor, required)
                Input tensor of Completion Network.
        - output (torch.Tensor, required)
                Output tensor of Completion Network.
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network.
    * returns:
                Image tensor inpainted using poisson image editing method.
    """
    num_samples = input.shape[0]
    ret = []
    mask = torch.cat((mask, mask, mask), dim=1)  # convert to 3-channel format


    # convert torch array to numpy array followed by
    # converting 'channel first' format to 'channel last' format.
    input_np = np.transpose(np.copy(input.cpu().numpy()), axes=(0, 2, 3, 1))
    output_np = np.transpose(np.copy(output.cpu().numpy()), axes=(0, 2, 3, 1))
    mask_np = np.transpose(np.copy(mask.cpu().numpy()), axes=(0, 2, 3, 1))

    # apply poisson image editing method for each input/output image and mask.
    for i in range(num_samples):
        inpainted_np = newblend(input_np[i], output_np[i], mask_np[i])
        inpainted = torch.from_numpy(np.transpose(inpainted_np, axes=(2, 0, 1)))
        inpainted = torch.unsqueeze(inpainted, dim=0)
        ret.append(inpainted)
    ret = torch.cat(ret, dim=0)
    return ret

def newblend(input, output, mask):
    foreground = output
    background = input
    alpha = mask
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    outImage = cv2.add(foreground, background)
    return outImage

class Februusmodel(nn.Module):
    def __init__(self, args):
        super(Februusmodel, self).__init__()
        self.base_model = generate_cls_model(model_name=args.model, num_classes=args.num_classes)
        self.base_model.load_state_dict(result['model'])
        self.base_model.eval()
        self.inpaint_model = CompletionNetwork()

        checkpoint = args.dataset + "_inpainting"
        msg = self.inpaint_model.load_state_dict(torch.load(checkpoint, map_location='cuda'))
        self.inpaint_model.eval()
        print(msg)
        self.data_mean, self.data_std = get_dataset_norm_stats(args.dataset)
        self.data_mean = torch.tensor(self.data_mean).reshape(1,3,1,1).to(args.device)
        self.data_std = torch.tensor(self.data_std).reshape(1,3,1,1).to(args.device)
        self.data_sz = (args.input_height,args.input_width)
        self.target_layers = [eval('self.base_model.{}'.format(args.cam_layer))]
        self.gcam = eval(args.cam_method)(model=self.base_model, target_layers=self.target_layers, use_cuda=True)
        self.MASK_COND = args.MASK_COND
        self.device = args.device

        self.mpv = self.data_mean
        if args.dataset == 'gtsrb':
            self.mpv = torch.tensor([0.33373367140503546, 0.3057189632961195, 0.316509230828686]).to(args.device)
            self.mpv = self.mpv.view(1,3,1,1)
        elif args.dataset == 'VGGFace2':
            self.mpv = torch.tensor([0.5, 0.5, 0.5]).to(args.device)
            self.mpv = self.mpv.view(1, 3, 1, 1)

    def normalize(self, x, mean, std):
        return (x-mean.to(x.device)) / std.to(x.device)

    def unnormalize(self, x, mean, std):
        return (x * std.to(x.device) + mean.to(x.device))

    def data_normalize(self, x):
        return self.normalize(x, self.data_mean, self.data_std)

    def data_unnormalize(self, x):
        return self.unnormalize(x, self.data_mean, self.data_std)

    def forward(self, x):
        outputs = self.base_model(x)
        return outputs


    def forward_inpaint(self, images):
        cleanimgs = []
        maskedimages = []
        # GAN inpainted
        # This is to apply Grad CAM to the load images
        # --------------------------------------------
        for j in range(len(images)):
            image = images[j]
            # image = self.februus_unnormalize(image)  # unnormalize to [0 1] to feed into GAN
            image = torch.unsqueeze(image, 0)  # unsqueeze meaning adding 1D to the tensor

            mask = self.gcam(image)  # get the mask through GradCAM


            cond_mask = mask >= self.MASK_COND
            mask = cond_mask.astype(int)

            # ---------------------------------------
            mask = np.expand_dims(mask, axis=0)  # add 1D to mask
            # mask = np.expand_dims(mask, axis=0)
            mask = torch.tensor(mask)  # convert mask to tensor 1,1,32,32
            mask = mask.type(torch.FloatTensor)
            mask = mask.to(self.device)
            x = self.data_unnormalize(image)  # original test image

            # inpaint
            with torch.no_grad():
                x_mask = x - x * mask + self.mpv * mask  # generate the occluded input [0 1]
                inputx = torch.cat((x_mask, mask), dim=1)
                output = self.inpaint_model(inputx)  # generate the output for the occluded input [0 1]

                # image restoration
                inpainted = poisson_blend_old(x_mask, output, mask)  # this is GAN output [0 1]
                inpainted = inpainted.to(self.device)

                # store GAN output
                cleanimgs.append(inpainted)
                maskedimages.append(x_mask)

        maskedimages = torch.cat(maskedimages)
        maskedimages = self.data_normalize(maskedimages)
        masked_outputs = self.base_model(maskedimages)

        # this is tensor for GAN output
        cleanimgs = torch.cat(cleanimgs)
        cleanimgs = self.data_normalize(cleanimgs)

        GAN_outputs = self.base_model(cleanimgs)
        return (masked_outputs, GAN_outputs)


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

    #set the parameter for the ft defense
    parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')

    parser.add_argument('--inpaint_arch', type=str, default='inpaint_vit_base_patch16')
    parser.add_argument('--inpaint_ckp', type=str, default='inpaint_visualize_vit_base.pth')
    parser.add_argument('--inpaint_num', type=int, default=1)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--MASK_COND', type=float, default=None)

    parser.add_argument('--cam_layer', type=str, default='layer4[-1]')
    parser.add_argument('--cam_method', type=str, default='GradCAM')

    arg = parser.parse_args()

    print(arg)
    return arg


def defense(args, result,):
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

    model = Februusmodel(args=args)
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
        poison_idx=np.zeros(len(data_bd_test)),
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
        poison_idx=np.zeros(len(data_clean_test)),
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=False,pin_memory=True)


    model.eval()
    asr_acc = 0
    for i, (inputs, labels) in enumerate(data_bd_loader):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model(inputs)
        pre_label = torch.max(outputs, dim=1)[1]
        asr_acc += torch.sum(pre_label == labels)
    original_ASR = asr_acc.item() / len(data_bd_loader.dataset) * 100

    clean_acc = 0
    for i, (inputs, labels) in enumerate(data_clean_loader):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model(inputs)
        pre_label = torch.max(outputs, dim=1)[1]
        clean_acc += torch.sum(pre_label == labels)
    original_ACC = clean_acc.item() / len(data_clean_loader.dataset) * 100

    logging.info("original ACC is {} and original ASR is {}".format(original_ACC, original_ASR))

    M = 2
    clean_acc = [0] * M
    cnt = [0] * M
    for i, (inputs, labels) in enumerate(data_clean_loader):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model.forward_inpaint(inputs)
        for m in range(M):
            pre_label = torch.max(outputs[m].detach().cpu(), dim=1)[1]
            clean_acc[m] += torch.sum(pre_label == labels.cpu())
            cnt[m] += len(labels)
    for m in range(M):
        inpaint_ACC = clean_acc[m].item() / len(data_clean_loader.dataset) * 100
        logging.info("ACC w/ Februus is {} for method {}".format(inpaint_ACC, m))

    asr_acc = [0] * M
    cnt = [0] * M
    all_preds = [[] for _ in range(M)]
    for i, (inputs, labels) in enumerate(data_bd_loader_org):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model.forward_inpaint(inputs)
        for m in range(M):
            pre_label = torch.max(outputs[m].detach().cpu(), dim=1)[1]
            asr_acc[m] += torch.sum(pre_label == labels.cpu())
            cnt[m] += len(labels)
            all_preds[m].append(pre_label.detach())
    for m in range(M):
        inpaint_ACC_org = asr_acc[m].item() / len(data_bd_loader_org.dataset) * 100
        logging.info("ACC on BAD IMAGES w/ Februus is {} for method {}".format(inpaint_ACC_org, m))

    all_labels = []
    for i, (inputs, labels) in enumerate(data_bd_loader):
        all_labels.append(labels)
    all_labels = torch.cat(all_labels)
    for m in range(M):
        inpaint_ASR = (torch.cat(all_preds[m])==all_labels).float().mean().item() * 100
        logging.info("ASR w/ Februus is {} for method {}".format(inpaint_ASR, m))



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

    ### 3. ft defense:
    defense(args, result)

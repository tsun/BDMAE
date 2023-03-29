# idea: select model you use in training and the trainer (the warper for training process)

import sys, logging
sys.path.append('../../')

import torch 
import torchvision.models as models
from typing import Optional

from utils.trainer_cls import ModelTrainerCLS

#trainer is cls
def generate_cls_model(
    sparsity: list = [0.]*100,
    model_name: str = 'resnet18',
    num_classes: int = 10,
    **kwargs,
):
    '''
    # idea: aggregation block for selection of classifcation models
    :param model_name:
    :param num_classes:
    :return:
    '''


    if model_name == 'resnet18':
        from models.resnet_comp import resnet18
        net = resnet18(num_classes=num_classes, **kwargs)
    elif model_name == 'stdresnet18':
        from torchvision.models.resnet import resnet18
        net = resnet18(num_classes=num_classes, **kwargs)
    elif model_name == 'pretrresnet18':
            from torchvision.models.resnet import resnet18
            net = resnet18(pretrained=True, **kwargs)
            net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        net = models.vgg16(num_classes=num_classes, **kwargs)
    elif model_name == 'pretrvgg16':
        net = models.vgg16(pretrained=True, **kwargs)
        net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, num_classes)
    elif model_name == 'Februus_cifar10':
        from models.februus import Februus_cifar10
        net = Februus_cifar10(128)
    elif model_name == 'Februus_GTSRB':
        from models.februus import Februus_GTSRB
        net = Februus_GTSRB(128)
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')

    return net

def generate_cls_trainer(
        model,
        attack_name: Optional[str] = None,
        amp: bool = False,
):
    '''
    # idea: The warpper of model, which use to receive training settings.
        You can add more options for more complicated backdoor attacks.

    :param model:
    :param attack_name:
    :return:
    '''

    trainer = ModelTrainerCLS(
        model=model,
        amp=amp,
    )

    return trainer
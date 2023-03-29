# BDMAE
Pytorch implementation of BDMAE. 
> [Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder](https://arxiv.org/abs/2303.15564)                 
> Tao Sun, Lu Pang, Chao Chen and Haibin Ling                 

## Abstract
Deep neural networks are vulnerable to backdoor attacks, where an adversary maliciously manipulates the model behavior through overlaying images with special triggers. Existing backdoor defense methods often require accessing a few validation data and model parameters, which are impractical in many real-world applications, e.g., when the model is provided as a cloud service. In this paper, we address the practical task of blind backdoor defense at test time, in particular for black-box models. The true label of every test image needs to be recovered on the fly from the hard label predictions of a suspicious model. The heuristic trigger search in image space, however, is not scalable to complex triggers or high image resolution. We circumvent such barrier by leveraging generic image generation models, and propose a framework of Blind Defense with Masked AutoEncoder (BDMAE). It uses the image structural similarity and label consistency between the test image and MAE restorations to detect possible triggers. The detection result is refined by considering the topology of triggers. We obtain a purified test image from restorations for making prediction. Our approach is blind to the model architectures, trigger patterns or image benignity. Extensive experiments on multiple datasets with different backdoor attacks validate its effectiveness and generalizability.  

### Blind Backdoor Defense at Test-time
<div align="center">
<img src="fig/task.png" width="60%"> <br>
</div>

### Framework of BDMAE
<div align="center">
<img src="fig/framework.png" width="60%"> <br>
</div>

## Usage
### Prerequisites
We experimented with python==3.8, pytorch==1.8.0, cudatoolkit==11.1. 

### Data Preparation

### Attack and Defense
```shell
# download MAE checkpoint
!wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth
!wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth

# generate backdoor attacked models
bash run/run_attack.sh

# conduct blind test-time defense with BDMAE
bash run/run_defense.sh
```

## Acknowledgements
Backdoor attack is adapted from [BackdoorBench](https://github.com/SCLBD/BackdoorBench).

MAE-related code is taken from [MAE](https://github.com/facebookresearch/mae).

SSIM calculation is based on [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim).

## Citation
If you find our paper and code useful for your research, please consider citing

```bibtex
@article{sun2023mask,
    author  = {Sun, Tao and Pang, Lu and Chen, Chao and Ling, Haibin},
    title   = {Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder},
    journal = {arXiv preprint arXiv:2303.15564 xx},
    year    = {2023}
}
```
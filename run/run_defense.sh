# cifar10 --------------------------------------------------

## badnet attack
CUDA_VISIBLE_DEVICES=0 python ./defense/BDMAE.py --result_file badnet_resnet18_cifar10_3by3_color --yaml_path ./config/defense/ft/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --log /log/cifar10/BDMAE/
CUDA_VISIBLE_DEVICES=0 python ./defense/BDMAE.py --result_file badnet_resnet18_cifar10_3by3_color --yaml_path ./config/defense/ft/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --log /log/cifar10/BDMAE_large/  --mae_arch mae_vit_large_patch16 --mae_ckp mae_visualize_vit_large_ganloss.pth

CUDA_VISIBLE_DEVICES=0 python ./defense/ft/Februus.py --result_file badnet_resnet18_cifar10_3by3_color --yaml_path ./config/defense/ft/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --log /log/cifar10/Februus_0.6_l3_XGradCAM/  --MASK_COND 0.6 --cam_layer 'layer3[-1]' --cam_method XGradCAM
CUDA_VISIBLE_DEVICES=0 python ./defense/ft/Februus.py --result_file badnet_resnet18_cifar10_3by3_color --yaml_path ./config/defense/ft/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --log /log/cifar10/Februus_0.6_l3_XGradCAM/  --MASK_COND 0.6 --cam_layer 'layer3[-1]' --cam_method GradCAMPlusPlus

## IAB attack
CUDA_VISIBLE_DEVICES=0 python ./defense/BDMAE.py --result_file IAB_resnet18_cifar10 --yaml_path ./config/defense/ft/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --log /log/cifar10/BDMAE/
CUDA_VISIBLE_DEVICES=0 python ./defense/BDMAE.py --result_file IAB_resnet18_cifar10 --yaml_path ./config/defense/ft/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --log /log/cifar10/BDMAE_large/  --mae_arch mae_vit_large_patch16 --mae_ckp mae_visualize_vit_large_ganloss.pth

CUDA_VISIBLE_DEVICES=0 python ./defense/ft/Februus.py --result_file IAB_resnet18_cifar10 --yaml_path ./config/defense/ft/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --log /log/cifar10/Februus_0.6_l3_XGradCAM/  --MASK_COND 0.6 --cam_layer 'layer3[-1]' --cam_method XGradCAM
CUDA_VISIBLE_DEVICES=0 python ./defense/ft/Februus.py --result_file IAB_resnet18_cifar10 --yaml_path ./config/defense/ft/cifar10.yaml --dataset cifar10 --model resnet18 --num_workers 0 --log /log/cifar10/Februus_0.6_l3_XGradCAM/  --MASK_COND 0.6 --cam_layer 'layer3[-1]' --cam_method GradCAMPlusPlus

# GTSRB --------------------------------------------------

## badnet attack
CUDA_VISIBLE_DEVICES=0 python ./defense/BDMAE.py --result_file badnet_resnet18_gtsrb_3by3_color --yaml_path ./config/defense/ft/gtsrb.yaml --dataset gtsrb --model resnet18 --num_workers 0 --log /log/gtsrb/BDMAE/
CUDA_VISIBLE_DEVICES=0 python ./defense/BDMAE.py --result_file badnet_resnet18_gtsrb_3by3_color --yaml_path ./config/defense/ft/gtsrb.yaml --dataset gtsrb --model resnet18 --num_workers 0 --log /log/gtsrb/BDMAE_large/  --mae_arch mae_vit_large_patch16 --mae_ckp mae_visualize_vit_large_ganloss.pth

CUDA_VISIBLE_DEVICES=0 python ./defense/ft/Februus.py --result_file badnet_resnet18_gtsrb_3by3_color --yaml_path ./config/defense/ft/gtsrb.yaml --dataset gtsrb --model resnet18 --num_workers 0 --log /log/gtsrb/Februus_0.6_l3_XGradCAM/  --MASK_COND 0.6 --cam_layer 'layer3[-1]' --cam_method XGradCAM
CUDA_VISIBLE_DEVICES=0 python ./defense/ft/Februus.py --result_file badnet_resnet18_gtsrb_3by3_color --yaml_path ./config/defense/ft/gtsrb.yaml --dataset gtsrb --model resnet18 --num_workers 0 --log /log/gtsrb/Februus_0.6_l3_XGradCAM/  --MASK_COND 0.6 --cam_layer 'layer3[-1]' --cam_method GradCAMPlusPlus

## IAB attack
CUDA_VISIBLE_DEVICES=0 python ./defense/BDMAE.py --result_file IAB_resnet18_gtsrb --yaml_path ./config/defense/ft/gtsrb.yaml --dataset gtsrb --model resnet18 --num_workers 0 --log /log/gtsrb/BDMAE/
CUDA_VISIBLE_DEVICES=0 python ./defense/BDMAE.py --result_file IAB_resnet18_gtsrb --yaml_path ./config/defense/ft/gtsrb.yaml --dataset gtsrb --model resnet18 --num_workers 0 --log /log/gtsrb/BDMAE_large/  --mae_arch mae_vit_large_patch16 --mae_ckp mae_visualize_vit_large_ganloss.pth

CUDA_VISIBLE_DEVICES=0 python ./defense/ft/Februus.py --result_file badnet_resnet18_gtsrb_3by3_color --yaml_path ./config/defense/ft/gtsrb.yaml --dataset gtsrb --model resnet18 --num_workers 0 --log /log/gtsrb/Februus_0.6_l3_XGradCAM/  --MASK_COND 0.6 --cam_layer 'layer3[-1]' --cam_method XGradCAM
CUDA_VISIBLE_DEVICES=0 python ./defense/ft/Februus.py --result_file badnet_resnet18_gtsrb_3by3_color --yaml_path ./config/defense/ft/gtsrb.yaml --dataset gtsrb --model resnet18 --num_workers 0 --log /log/gtsrb/Februus_0.6_l3_XGradCAM/  --MASK_COND 0.6 --cam_layer 'layer3[-1]' --cam_method GradCAMPlusPlus


# Cifar10 --------------------------------------------------

CUDA_VISIBLE_DEVICES=0 python ./attack/badnet_attack.py --yaml_path ../config/attack/badnet/cifar10.yaml --dataset cifar10 --dataset_path ../data --save_folder_name badnet_resnet18_cifar10_3by3_color --random_seed 0 --model resnet18 --attack_target 0 --patch_mask_path ../resource/badnet/cifar10_3by3_color.npy
CUDA_VISIBLE_DEVICES=0 python ./attack/inputaware_attack.py --yaml_path ../config/attack/inputaware/cifar10.yaml --dataset cifar10 --dataset_path ../data --save_folder_name IAB_resnet18_cifar10 --random_seed 0 --model resnet18 --attack_target 0 --epochs 200

# GTSRB --------------------------------------------------

CUDA_VISIBLE_DEVICES=0 python ./attack/badnet_attack.py --yaml_path ../config/attack/badnet/gtsrb.yaml --dataset gtsrb --dataset_path ../data --save_folder_name badnet_resnet18_gtsrb_3by3_color --random_seed 0 --model resnet18 --attack_target 0 --patch_mask_path ../resource/badnet/cifar10_3by3_color.npy
CUDA_VISIBLE_DEVICES=0 python ./attack/inputaware_attack.py --yaml_path ../config/attack/inputaware/gtsrb.yaml --dataset gtsrb --dataset_path ../data --save_folder_name IAB_resnet18_gtsrb --random_seed 0 --model resnet18 --attack_target 0 --epochs 200



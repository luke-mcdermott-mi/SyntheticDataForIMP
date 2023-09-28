Code is based on Efficient Dataset Condensation[https://github.com/snu-mllab/Efficient-Dataset-Condensation], see this for downloading precomputed distilled data.

# Synthetic Data and Iterative Magnitude Pruning: A Linear Mode Connectivity Study
Early notes for how to run things, a rework for this repository is coming soon.

run this to test distilled data for cifar-10
python test.py -d cifar10 -n convnet -f 2 --ipc 10 --repeat 3 --data_dir './results/'

To find the synthetic tickets, you need to run IMP with the distilled training: 
python prune.py -d cifar10 -f 2 --ipc 50 --data_dir './results/' --device 3
python prune.py -d imagenet -f 3 --ipc 20 --data_dir './results/' --device 5 --nclass 100

Once you get all the untrained subnetworks / synthetic tickets at initialization, train them on real data across seeds 0 & 1:
python train_pruned.py -d cifar10 --device 1 --data_dir './results/' --model resnet18_bn
python train_pruned.py -d imagenet --device 3 --data_dir './results/' --model resnet10_bn --ipc_path 10 --nclass 10

To check linear interpolation, you need to have the seed0_iter and seed1_iter models done. Then run 
python lmc.py -d cifar100 --device 3 --data_dir './results/' --model convnet3 --ipc_path 1
python lmc.py -d imagenet --nclass 10 --device 2 --data_dir './results/' --model resnet10_bn --ipc_path 10

To get lottery tickets with IMP run
python prune_imp.py -d cifar10 --device 6 --data_dir './results/' --model resnet18_bn
python prune_imp.py -d imagenet --nclass 100 --device 5 --data_dir './results/' --model resnet10_bn

To retrain the untrained imp-chosen subnetworks:
python train_imp.py -d cifar10 --device 1 --data_dir './results/' --model resnet18_bn --seed 4
python train_imp.py -d imagenet --nclass 10 --device 2 --data_dir './results/' --model resnet18_bn --seed 0

To check linear interpolation of imp subnetworks, you need to have the seed0_iterx_imp and seed1_iterx_imp models done. Then run 
python lmc_imp.py -d cifar100 --device 2 --data_dir './results/' --model resnet18_bn
python lmc_imp.py -d imagenet --nclass 10 --device 2 --data_dir './results/' --model resnet10_bn

To get the linear interpolation of the dense model at different points in training
python lmc_dense_training.py -d cifar10 --device 4 --data_dir './results/' --model resnet10_bn

To get the loss landscape of a model, first you need seeds0-4 trained, then run:
python multiple_lmc.py -d cifar10 --device 3 --data_dir './results/' --model resnet10_bn --prune_iter 4

To compute all the distances between your different trained seeds run:
python calc_model_distance.py -d cifar10 --device 3 --data_dir './results/' --model convnet3 --prune_iter 9

To get synthetic tickets after some pretraining, do
python prune_rewind.py -d cifar10 -f 2 --ipc 10 --data_dir './results/' --device 3

To linear interpolated just a single layer, you need to have the seed0_iter and seed1_iter models done. Then run 
python layerwise_lmc.py -d cifar10 --device 3 --data_dir './results/' --model convnet3

To get random subnetowkrs run
python random_prune.py -d cifar10 --device 3 --data_dir './results/' --model resnet18_bn

To get the loss landscapes for models trained on real data, but the corresponding spot on distilled data: 
python distilled_loss_landscape.py -d cifar10 -f 2 --ipc 10 --data_dir './results/' --device 3 --prune_iter 9

To get a list of sorted eigenvalues of the hessian for sharpness run this:
python calc_hessian.py -d cifar10 --device 3 --data_dir './results/' --model resnet18_bn --prune_iter 9

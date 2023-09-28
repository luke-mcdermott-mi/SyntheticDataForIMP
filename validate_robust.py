import torch
import torch.nn.utils.prune as prune
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from prune import get_parameters_to_prune, convnet3, resnet10_bn, resnet18_bn
import models.convnet as CN
import models.resnet as RN
import json
from torch.quantization import FakeQuantize

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class CorruptedDataset(Dataset):
    def __init__(self, corruption):
        path = '/home/ubuntu/luke/LMC/CIFAR-10C/CIFAR-10-C/'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.images = torch.tensor(np.load(path + corruption + '.npy')).to(torch.float).to(device) / 255
        self.images = self.images.permute(0,3,2,1)
        self.labels = torch.tensor(np.load(path + 'labels.npy')).to(device)

        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
        self.images = (self.images - mean) / std

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def test(model, test_loader):
  device = torch.device('cuda:7')
  # Test the model
  model.eval()
  model.to(device)
  with torch.no_grad():
      correct = 0
      total = 0
      for i, (images, labels) in enumerate(test_loader):
          images, labels = images.to(device), labels.to(device)
          test_output = model(images)
          pred_y = torch.max(test_output, 1)[1].data.squeeze()
          correct += (pred_y == labels).sum().item()
          total += labels.size(0)
      accuracy = correct / total

  print('Test Accuracy:', accuracy)
  return accuracy

def get_parameters_to_prune(model):
  parameters_to_prune = []
  for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
      parameters_to_prune.append((module, 'weight'))
  return tuple(parameters_to_prune)

seed = 0
device = torch.device('cuda:2')
corruptions = ['contrast']
"""
corruptions = ['pixelate', 'impulse_noise', 'contrast', 'motion_blur', 'gaussian_noise', 'snow', 'brightness',
               'saturate', 'frost', 'gaussian_blur', 'elastic_transform', 'defocus_blur', 'shot_noise', 'spatter',
               'glass_blur', 'speckle_noise', 'zoom_blur', 'jpeg_compression', 'fog']
"""

models = ['seed' + str(seed) + '_iter0', 'seed' + str(seed) + '_iter4','seed' + str(seed) + '_iter4_imp', 'seed' + str(seed) + '_iter9','seed' + str(seed) + '_iter9_imp']
accuracies = {}
path = '/home/ubuntu/luke/LMC/pruned_models/cifar_10/resnet-18/'



# Function to recursively apply fake quantization to layers of the model
def apply_fake_quantization(layer):
    for name, module in layer.named_children():
        if len(list(module.children())) > 0:
            apply_fake_quantization(module)
        
        # Apply fake quantization to the weight of Conv2d and Linear layers
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            module.weight = torch.nn.Parameter(fake_quant(module.weight))
            if module.bias is not None:
                module.bias = torch.nn.Parameter(fake_quant(module.bias))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

results = np.zeros((5,11)) #4 seeds, 5 models, no/yes quantization

for seed in range(2,5):
    print('seed: ', seed)

    models = ['seed' + str(seed) + '_iter0']
    for extension in ['', '_imp']:
        for i in range(1,11):
            models.append('seed' + str(seed) + '_iter' + str(i) + extension)
    
    #models = ['seed' + str(seed) + '_iter0', 'seed' + str(seed) + '_iter4','seed' + str(seed) + '_iter4_imp', 'seed' + str(seed) + '_iter9','seed' + str(seed) + '_iter9_imp']
    #models = ['seed0_iter4', 'seed0_iter9', 'seed0_iter4_imp', 'seed0_iter9_imp']
    models = ['seed' + str(seed) + '_iter' + str(i) for i in range(0,11)]

    for model_idx, model_name in enumerate(models):
        print(model_name)

        
        #for quant_idx, quant_max in enumerate([127,63]):
        """
        #for quant_idx, quant_max in enumerate([1024, 255, 127, 63, 31, 15, 7, 3]):
            print(quant_max)
            if quant_max < 1024:
                fake_quant = FakeQuantize(quant_min=0, quant_max=quant_max)
                #fake_quant = FakeQuantize()
        """
        #for i in range(1):
            #fake_quant = FakeQuantize()

        model = RN.ResNet('cifar10', 18, 10, 'batch', 1024, nch=3) #CN.ConvNet(100)
        prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)
        model.load_state_dict(torch.load(path + model_name))
        prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)
        prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)
        for module, name in get_parameters_to_prune(model):
            prune.remove(module, name)
       #if quant_max < 1024:
        #if i == 1:
        #apply_fake_quantization(model) #apply fake quantization
        model = model.to(device)
        acc = test(model, testloader)
        results[seed, model_idx] = acc
        #print('')



np.save('/home/ubuntu/luke/LMC/Quantization/rn18_c10_syn', results)

"""
for corruption in corruptions:
    print('Corruption:', corruption)
    dataset = CorruptedDataset(corruption)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    for model_name in models:
        model = RN.ResNet('cifar10', 10, 10, 'batch', 1024, nch=3) #CN.ConvNet(10).to(device) 
        prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)
        model.load_state_dict(torch.load(path + model_name))
        prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)
        prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)
        for module, name in get_parameters_to_prune(model):
            prune.remove(module, name)
        apply_fake_quantization(model) #apply fake quantization
        model = model.to(device)
        acc = test(model, dataloader)
        accuracies[ corruption + '.' + model_name ] = acc

    with open('/home/ubuntu/luke/LMC/CIFAR-10C/rn10_quantize_results.json', 'w') as f:
        json.dump(accuracies, f, indent=4)
"""

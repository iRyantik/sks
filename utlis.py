from __future__ import print_function, division

import torch
from torchvision import datasets, transforms
import time
import os
import json
import copy


# Dataset initialization
def data_loading(data_dir='data', PARAM=None):
    if PARAM == None:
        PARAM = {
            'batch_size': 4,
            'transform_train': (0.485, 0.456, 0.406), 
            'transfrom_test': (0.229, 0.224, 0.225),
        }

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(PARAM['transform_train'], (0.229, 0.224, 0.225))
        ]),
        'test': transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'test']} # Read train and test sets, respectively.

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=PARAM['batch_size'],
                                                shuffle=True, num_workers=4)
                for x in ['train', 'test']}

    trainloader = dataloaders['train']
    testloader = dataloaders['test']

    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    class_names = image_datasets['train'].classes

    return trainloader, testloader, class_names

def save_result_n_model(save_name, data, model):
    print('==========Storing==========')
    save_dir = os.path.join('results', save_name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("Result directory " , save_dir ,  " Created")
    else:    
        print("Result directory " , save_dir ,  
        " already exists and the data in there will be covered")

    result_file = os.path.join(save_dir, 'result.json')
    with open(result_file, 'w') as outfile:
        json.dump(data, outfile)
    
    PATH = os.path.join(save_dir, 'cifar_net.pth')
    torch.save(model.state_dict(), PATH)

    return 

def restore_model(save_name):
    save_dir = os.path.join('results', save_name)
    PATH = os.path.join(save_dir, 'cifar_net.pth')
    model = Net()
    model.load_state_dict(torch.load(PATH))
    return model
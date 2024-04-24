from __future__ import print_function, division

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import os
import json
import copy
import torch.nn as nn
import torch.nn.functional as F

from model import *
from utlis import *

import timeit


# Implement training and testing procedures
def train(model, criterion, optimizer, scheduler, num_epochs=25, save_name=None, if_eval=False):
    print("==========Start Training==========")
    start = timeit.default_timer()
    
    loss_list = []
    time_list = []
    test_accuracy = []
    test_class_accuracy = {}
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        ep_start = timeit.default_timer()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1000))
                epoch_loss += running_loss
                running_loss = 0.0
        scheduler.step()
        loss_list.append(epoch_loss)

        ep_time = timeit.default_timer() - ep_start
        time_list.append(ep_time)
        print('Epoch Time: ', ep_time)  

        if if_eval:
            print('====In-trianing Evaluation====')
            test_start = timeit.default_timer()
            epoch_test_accuracy, epoch_test_class_accuracy = test(model, 
                                    if_print_total=True, if_print_class=False)
            test_accuracy.append(epoch_test_accuracy)
            for key, value in epoch_test_class_accuracy.items():
                if key in test_class_accuracy.keys():
                    test_class_accuracy[key].append(value)
                else:
                    test_class_accuracy[key] = [value]
            
            test_time = timeit.default_timer() - test_start
            print('Test Time: ', test_time)  
   
    print('==========Finished Training==========')
    train_time = timeit.default_timer() - start
    print('Total Training Time: ', train_time)

    if save_name is not None:
        data = {}
        data['PARAM'] = PARAM
        data['loss'] = loss_list
        data['time'] = time_list
        if if_eval:
            data['test_accuracy'] = test_accuracy
            data['test_class_accuracy'] = test_class_accuracy
        save_result_n_model(save_name, data, model)
    
    return model

def test(model, if_print_total=True, if_print_class=True):
    if if_print_class:
        print('==========Evaluation on Test==========')

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # total
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # class-wise
            for label, predicted in zip(labels, predicted):
                if label == predicted:
                    correct_pred[class_names[label]] += 1
                total_pred[class_names[label]] += 1

    # total        
    test_accuracy = correct / total
    if if_print_total:
        print('Accuracy of the network on the test images: %d %%' % (100 * test_accuracy))
    
    # class-wise
    test_class_accuracy = dict()
    for classname, correct_count in correct_pred.items():
        accuracy = float(correct_count) / total_pred[classname]
        test_class_accuracy[classname] = accuracy
        if if_print_class:
            print("Accuracy for class {:5s} is: {:.1f} %".format(classname, 100 * accuracy))

    return test_accuracy, test_class_accuracy

if __name__ == '__main__':

    # You can change these data augmentation and normalization strategies for
    # better training and testing (https://pytorch.org/vision/stable/transforms.html)

    # Adjust the following hyper-parameters: learning rate, decay strategy, number of training epochs.
    PARAM = {
        "lr": 1e-4,
        'step_size': 20, 
        'gamma': 0.1,
        'num_epochs': 25,
        'batch_size': 4,
        'transform_train': (0.485, 0.456, 0.406), 
        'transfrom_test': (0.229, 0.224, 0.225),

        'save_name': 'saved_models/model1.pth',
        'if_eval': True,
    }

    trainloader, testloader, class_names = data_loading(PARAM=PARAM)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Set device to "cpu" if you have no gpu
    model_ft = Net() # Model initialization
    model_ft = model_ft.to(device) # Move model to cpu
    criterion = nn.CrossEntropyLoss() # Loss function initialization

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=PARAM['lr']) # Optimizer initialization
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, 
                            step_size=PARAM['step_size'], gamma=PARAM['gamma']) # Learning rate decay strategy

    model_trained = train(model_ft, criterion, optimizer_ft, exp_lr_scheduler, 
                            num_epochs=PARAM['num_epochs'], save_name=PARAM['save_name'], if_eval=PARAM['if_eval'])
    
    if not PARAM['if_eval'] or not PARAM['save_name']:
        test(model_trained)
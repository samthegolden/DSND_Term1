from argparse import ArgumentParser

import sys
import os
sys.path.append("/Users/silvas/training/udacity/ml_intro_pytorch/github/DSND_Term1/projects/p2_image_classifier")
#print (sys.path)

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import numpy as np
from numpy import ndarray
import torch

import matplotlib.pyplot as plt

###

from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models

data_dir = 'flowers'
train_dir = data_dir + '/train'
train_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)


def save_checkpoint(model, filepath):
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    model.class_to_idx = train_datasets.class_to_idx
    '''
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.epochs = epochs
    model.optimizer_state_dict = optimizer.state_dict
    
    checkpoint = {'input_size': 25088,
          'output_size': 102,
          'hidden_layers': [each.out_features for each in model.hidden_layers],
          'state_dict': model.state_dict() # includes parameters set before 'checkpoint' ?
            }
          
    '''
    checkpoint = {
    'input_size': 25088,
    'output_size': 102,
    'state_dict': model.state_dict(),
    'mapping':    train_datasets.class_to_idx,
    'classifier': model.classifier
    }
    torch.save(model, filepath)

###
def train(data_dir, save_dir="checkpoint.pth", arch="vgg16", lr=0.0001, hidden_units=5000, epochs=5, gpu=True):
    #data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #When you're testing however, you'll want to use images that aren't altered other than normalizing. So, for validation/test images, you'll typically just resize and crop.
    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)


    # LOAD PRETRAINED MODEL
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)

    device = torch.device("cuda") if gpu else torch.device("cpu")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    ## BUILD CLASSIFIER

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict(
                [
                    ('fc1', nn.Linear(25088, hidden_units)),
                    ('relu', nn.ReLU()),
                    ('drop', nn.Dropout(0.2)),
                    ('fc4', nn.Linear(hidden_units, 102)),
                    ('output', nn.LogSoftmax(dim=1))
                ]))
            
    model.classifier = classifier
    #print(model.classifier.in_features)
    #print(model.fc.in_features)
    #print(model.classifier)

    features = list(model.classifier.children())[:-1]
    #num_filters = model.classifier[len(features)].in_features

    criterion = nn.NLLLoss()
    # Optimize only classifier parameters, not model features parameters
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr) # 0.002 (yes, 20x bigger) was causing accuracy fluctuation on validation set

    model.to(device)

    '''
    # Only train the classifier parameters, feature parameters are frozen
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = optim.Adam(params_to_update, lr = 0.001)
    
            
    for images, labels in trainloader:
        print(images.shape) # torch.Size([32, 3, 224, 224]) So, 32 images per batch, 3 color channel, and 224x224 images.
        print(labels.shape) # torch.Size([32])
        break
            
    image, label = next(iter(trainloader))
    helper.imshow(image[0,:]);
    helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')
    '''

    # TRAIN

    epochs = 2
    steps = 0
    running_loss = 0
    print_every = 10
    

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # images = images.view(images.shape[0], -1) # no need of reshaping...why?
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # outputs = model.forward(inputs)
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Validation loop -but then why use testloader and not validloader??
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                with torch.no_grad():
                    model.eval()
                    for inputs, validlabels in validloader:
                        inputs, validlabels = inputs.to(device), validlabels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, validlabels)
                        
                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == validlabels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {e+1}/{epochs}.. " \
                    f"Train loss: {running_loss/print_every:.3f}.. " \
                    f"Test loss: {test_loss/len(validloader):.3f}.. " \
                    f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
        '''    
        else:
            print(f"Training loss: {running_loss} ")
        ''' 

    save_checkpoint(model, save_dir)
###

if __name__ == "__main__":

    ap = ArgumentParser(description="ML")
    ap.add_argument("data_directory", help="The data directory that feeds the script.")
    ap.add_argument('--save_dir', help="Save data to dir.")
    ap.add_argument('--arch', help="Architecture.")
    ap.add_argument("--learning_rate", type=float, help="Learning rate")
    ap.add_argument('--hidden_units', type=int, help="Hidden units")
    ap.add_argument('--epochs', type=int, help="Epochs")
    ap.add_argument('--gpu', action='store_true', help="GPU")
    
    args = ap.parse_args()
    
    '''
    print(args.data_directory)
    print(args.save_dir)
    print(args.arch)
    print(args.learning_rate)
    print(args.hidden_units)
    print(args.epochs)
    print(args.gpu)
    '''

    kwargs = dict(save_dir=args.save_dir, arch=args.arch, lr=args.learning_rate, hidden_units=args.hidden_units, epochs=args.epochs, gpu=args.gpu)
    # only pass parameters that are not None
    train(args.data_directory, **{k: v for k, v in kwargs.items() if v is not None})

    # python train.py flowers --gpu
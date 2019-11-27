import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}


def load_data(where  = "./flowers" ):
    '''
    Arguments : the datas' path
    Returns : The loaders for the train, validation and test datasets
    This function receives the location of the image files, applies the necessery transformations (rotations,flips,normalizations and crops) and converts the images to tensor in order to be able to be fed into the neural network
    '''

    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Apply the required transfomations to the test dataset in order to maximize the efficiency of the learning
    #process


    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Crop and Resize the data and validation images in order to be able to be fed into the network

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    image_datasets = [train_data,validation_data,test_data]

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # The data loaders are going to use to load the data to the NN(no shit Sherlock)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    vloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)
    dataloaders=[trainloader,vloader,testloader]



    return image_datasets,dataloaders



def nn_setup(structure='densenet121',dropout=0.5, hidden_layer1 = 120,lr = 0.001,power='gpu'):
    '''
    Arguments: The architecture for the network(alexnet,densenet121,vgg16), the hyperparameters for the network (hidden layer 1 nodes, dropout and learning rate) and whether to use gpu or not
    Returns: The set up model, along with the criterion and the optimizer fo the Training
    '''

    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Im sorry but {} is not a valid model.Did you mean vgg16,densenet121,or alexnet?".format(structure))

    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('dropout',nn.Dropout(dropout)),
        ('inputs', nn.Linear(arch[structure], hidden_layer1)),
        ('relu1', nn.ReLU()),
        ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
        ('relu2',nn.ReLU()),
        ('hidden_layer2',nn.Linear(90,80)),
        ('relu3',nn.ReLU()),
        ('hidden_layer3',nn.Linear(80,102)),
        ('output', nn.LogSoftmax(dim=1))
                          ]))


    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr )


    if torch.cuda.is_available() and power == 'gpu':
        model.cuda()
        
    return model,optimizer,criterion




def train_network(model,optimizer,criterion,dataloaders,epochs=5,power='gpu'):
    '''
    Arguments: The model, the criterion, the optimizer, the number of epochs, teh dataset, and whether to use a gpu or not
    Returns: Nothing
    This function trains the model over a certain number of epochs and displays the training,validation and accuracy every "print_every" step using cuda if specified. The training method is specified by the criterion and the optimizer which are NLLLoss and Adam respectively
    '''
    steps = 0
    running_loss = 0

    print("--------------Training starts------------- ")

    for e in range(epochs):

        train_mode = 0
        valid_mode =1

        for mode in[train_mode,valid_mode]:
            if mode==train_mode:
                model.train()
            else:
                model.eval()
            
            pass_count=0

            for data in dataloaders[mode]:
                pass_count +=1
                steps +=1
                inputs,labels=data
                if torch.cuda.is_available() and power == 'gpu':
                    inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())
                else:
                    inputs, labels=Variable(inputs),Variable(labels)
                
                optimizer.zero_grad()
                output = model.forward(inputs.cuda())
                loss = criterion(output,labels)

                if mode==train_mode:
                    loss.backward()
                    optimizer.step()
                running_loss +=loss.item()
                ps = torch.exp(output).data
                equality = (labels.data==ps.max(1)[1])
                accuracy= equality.type_as(torch.FloatTensor()).mean()

            if mode==train_mode:
                print("\nEpoch: {}/{} ".format(e+1,epochs),
                         "\nTraing Loss: {:.4f} ".format(running_loss/pass_count))
            else:
                print("Validation Loss: {:.4f} ".format(running_loss/pass_count),
                      "Accuracy: {: 4f}".format(accuracy))
            running_loss = 0
            
    model.eval()

    for data in dataloaders[2]:
        pass_count +=1
        accuracy = 0
        inputs,labels=data
        if torch.cuda.is_available() and power == 'gpu':
            inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())
        else:
            inputs, labels=Variable(inputs),Variable(labels)
                
        output = model.forward(inputs.cuda())
        ps = torch.exp(output).data
        equality = (labels.data==ps.max(1)[1])
        accuracy= equality.type_as(torch.FloatTensor()).mean()     
            
    print("Testing Accuracy : {:.4f}".format(accuracy/pass_count))

    print("-------------- Finished training -----------------------")
 


def save_checkpoint(model,image_datasets,path='checkpoint.pth',structure ='densenet121', hidden_layer1=120,dropout=0.5,lr=0.001,epochs=5):
    '''
    Arguments: The saving path and the hyperparameters of the network
    Returns: Nothing
    This function saves the model at a specified by the user path
    '''
    model.class_to_idx = image_datasets[0].class_to_idx
    
    torch.save({'structure' :structure,
                'hidden_layer1':hidden_layer1,
                'dropout':dropout,
                'lr':lr,
                'nb_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)


def load_checkpoint(path='checkpoint.pth'):
    '''
    Arguments: The path of the checkpoint file
    Returns: The Neural Netowrk with all hyperparameters, weights and biases
    '''
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']

    model,_,_ = nn_setup(structure , dropout,hidden_layer1,lr)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model


def process_image(image):
    '''
   Scales,crops and normalizes a PIL for a pytorch model and return a Numpy Array
    '''

    img_pil = Image.open(image)
    
    adjustments = transforms.Compose([ 
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = adjustments(img_pil)

    return img_tensor


def predict(image_path,model, topk=5,power='gpu'):
    '''
    Arguments: The path to the image, the model, the number of prefictions and whether cuda will be used or not
    Returns: The "topk" most probable choices that the network predicts
    '''

    if torch.cuda.is_available() and power == 'gpu':
        model.eval()

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)


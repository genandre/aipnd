import matplotlib.pyplot as plt

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchvision import transforms, datasets, models

from PIL import Image

import json

import pdb

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: dropout probability
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
        self.input_size = input_size
        self.output_size = output_size
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
            x = self.output(x)
        
        return F.log_softmax(x, dim=1)


def validation(model, testloader, criterion, use_gpu):
    correct = 0
    total_images = 0
    test_loss = 0
    for data in testloader:
        images, labels = data
        if(use_gpu):
            images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model.forward(images)
        #calculate test loss
        test_loss += criterion(outputs, labels).item()
        #calculate accuracy
        _,predictions = torch.max(outputs.data, 1)
        correct+= (predictions == labels).sum().item()
        #calculate total images
        total_images+= labels.size(0)
        
    accuracy = (100 * (correct / total_images))
    test_loss = test_loss / total_images

    return test_loss, accuracy, total_images



def train(model, trainloader, testloader, criterion, optimizer, epochs, print_every, use_gpu):
    steps = 0
    
    if(use_gpu): 
        model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        model.train()
        for ii, (inputs,labels) in enumerate(trainloader):
            steps += 1
            if(use_gpu):
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                    # Model in inference mode, dropout is off
                    model.eval()

                    # Turn off gradients for validation, will speed up inference
                    with torch.no_grad():
                        test_loss, accuracy, total_images = validation(model, testloader, criterion, use_gpu)

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(test_loss),
                          "Validation Accuracy: {:.3f}".format(accuracy),
                          "Total images validated: {}".format(total_images))

                    running_loss = 0

                    # Make sure dropout and grads are on for training
                    model.train()

                    
def save_checkpoint(pretrained_model_name, model, filepath):
    checkpoint = { 'pretrained_model_name' : pretrained_model_name,
                   'input_size' : model.classifier.input_size,
                   'output_size' : model.classifier.output_size,
                   'hidden_layers' : [each.out_features for each in model.classifier.hidden_layers],
                   'dropout' : model.classifier.dropout,
                   'state_dict' : model.state_dict(),
                   'class_to_idx' : get_cat_to_name()}
    torch.save(checkpoint, filepath)

    
def get_cat_to_name(cat_to_name='cat_to_name.json'):
    with open(cat_to_name, 'r') as f:
        return json.load(f)
    
    
def load_checkpoint(filepath, use_gpu):
    if(use_gpu):
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    model = eval('models.' + checkpoint['pretrained_model_name'] + '(pretrained=True)')
    model.classifier = Network(checkpoint['input_size'],
                        checkpoint['output_size'],
                        checkpoint['hidden_layers']
                       )
    model.classifier.dropout = checkpoint['dropout']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns tensor
    '''
    pil_image = Image.open(image)
    image_transform = transforms.Compose([transforms.Resize(255),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    return image_transform(pil_image)

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk, use_gpu):
    image = process_image(image_path)
    #Insert dimension for batch size
    if(use_gpu):
        image = image.unsqueeze_(0).to('cuda')
    else:
        image = image.unsqueeze_(0).to('cpu')
    
    model.eval()
    
    if(use_gpu): 
        model.to('cuda')
    
    with torch.no_grad():
        outputs = model.forward(image)
    #calculate accuracy
    log_softmax_probs , predictions = torch.topk(outputs.data, k=topk, dim=1)
    probabilities = torch.exp(log_softmax_probs)
    return probabilities.cpu().numpy()[0], predictions.cpu().numpy()[0]

def print_predictions(probs, classes, cat_to_name):
    class_names = [get_cat_to_name(cat_to_name).get(str(i)) for i in classes ]
    for class_name, prob in zip(class_names,probs):
        print('{} : {}'.format(class_name, prob))

def show_prediction_chart(probs, classes, image):
    class_names = [get_cat_to_name().get(str(i)) for i in classes ]
    
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(13,13))
    
    #subplot 1
    ax1 = imshow(process_image(image),ax1)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    
    #subplot 2
    
    plt.sca(ax2)
    index = np.arange(len(class_names))
    plt.bar(index, probs)
    plt.xlabel('Flower', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.xticks(index, class_names, fontsize=12, rotation=30)

    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    ax2.set_aspect(asp)
    
    plt.show()    
    
import argparse
import nn_model

import torch
from torch import nn
from torch import optim

from torchvision import transforms, datasets, models

import os 
from workspace_utils import active_session

#for debugging
#import pdb

def main():
    # Get input arguments
    in_args = get_input_args()
    
    # Define training, validation and testing directories
    train_dir = os.path.join(in_args.data_dir, 'train')
    valid_dir = os.path.join(in_args.data_dir, 'valid')
    test_dir = os.path.join(in_args.data_dir, 'test')
    
    # Define training, validation and testing transforms 
    train_transforms = transforms.Compose([transforms.RandomRotation(90),
                                       transforms.RandomHorizontalFlip(),
                                      transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([ transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([ transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    # Create training, validation and testing datasets
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    # Define data loaders 
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size =64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size =32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size =32, shuffle=True)
    
    # Create model
    model = eval('models.' + in_args.arch + '(pretrained=True)')
    
    #Freeze pre-trained model for training
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn_model.Network(get_classifier_in_features(model), 102, in_args.hidden_units, drop_p=0.5)
    
    # Define criterion and optimizer along with some other hyperparams
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), in_args.learning_rate)
    epochs = in_args.epochs
    print_every = 40
     
    with active_session():
        #train model
        print('Training model now...')
        nn_model.train(model, trainloader, validloader, criterion, optimizer, epochs, print_every, in_args.gpu)

        #test model
        print('Testing model now...')
        if(in_args.gpu): 
            model.to('cuda')
        model.eval()
        with torch.no_grad():
            _, accuracy, total_images = nn_model.validation(model, testloader, criterion, in_args.gpu)
            print("Accuracy of network on {} images = {:.4f}".format(total_images, accuracy))
        

        #Test code
        total_images =  0
        correct = 0 
        for dirpath, dirnames, filenames in os.walk('flowers/test'):
            for filename in filenames:
                total_images+=1
                input_image_path = os.path.join(dirpath, filename)
                true_label = dirpath.split(os.sep)[2]
                probs, classes = nn_model.predict(input_image_path, model, 1, True)
                predicted = nn_model.get_cat_to_name().get(str(classes[0]))
                actual = nn_model.get_cat_to_name().get(true_label)    
                if actual == predicted:
                    print('match prediction for {}'.format(actual))
                    correct+=1
        print('correct predictions = {}'.format(correct))
        print('total images = {}'.format(total_images))
        print('accuracy = {}'.format((correct / total_images) * 100.0))
        #

        
        #save model
        print('Saving trained model to {}'.format(in_args.save_dir))
        nn_model.save_checkpoint(in_args.arch, model, in_args.save_dir)
    

def allowed_models():
    return ['resnet18', 'alexnet', 'squeezenet1_0', 'vgg16', 'densenet161', 'inception_v3']

def get_classifier_in_features(model):
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            for step in model.classifier:
                if isinstance(step, nn.Linear): 
                    return step.in_features 
                elif isinstance(step, nn.Conv2d):
                    return step.in_channels
        elif isinstance(model.classifier, nn.Linear):
                    return model.classifier.in_features
    elif hasattr(model, 'fc'):
        if isinstance(model.fc, nn.Linear):
            return model.fc.in_features
    
    
def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_dir", type=str,
                   help="Path to the input image files")
    
    parser.add_argument("--save_dir", type=str,
                   help="Path to the files(default- 'saved_checkpoint/checkpoint.pth')",
                   default="saved_checkpoint/checkpoint.pth")
    
    parser.add_argument("--arch", type=str, choices = allowed_models(),
                   help="Pretrained model architecture to use for image classification (default- vgg16)",
                   default="vgg16")
    
    parser.add_argument("--learning_rate", type=float,
                   help="Learning rate for model (default- '0.001')",
                   default=0.001)
    
    parser.add_argument("--hidden_units", type=int, nargs='+',
                   help="List of number of hidden units to add to network. (default- '512')",
                   default=[512])
        
    parser.add_argument("--epochs", type=int,
                   help="Number of epochs for training (default- '20')",
                   default=20)
    
    parser.add_argument("--gpu", action="store_true",
                   dest='gpu', help="if passed, uses GPU for training",
                   default=False)
    
    return parser.parse_args()
        


if __name__ == "__main__":
    main()

    

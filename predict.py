import argparse
import nn_model

import torch
from torch import nn
from torch import optim

from torchvision import transforms, datasets, models

import pdb

def main():
    # Get input arguments
    in_args = get_input_args()
    
    #load model from checkpoint
    print('Loading model from saved checkpoint...')
    model = nn_model.load_checkpoint(in_args.checkpoint, in_args.gpu)
    
    #test code
    print('Testing model now...')
    test_dir = 'flowers' + '/test'
    test_transforms = transforms.Compose([ transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size =32, shuffle=True)
    criterion = nn.NLLLoss()
    if(in_args.gpu): 
        model.to('cuda')
    model.eval()
    with torch.no_grad():
        _, accuracy, total_images = nn_model.validation(model, testloader, criterion, in_args.gpu)
        print("Accuracy of network on {} images = {:.4f}".format(total_images, accuracy))
    #
    
    print('Predicting proabable name for {} input image/s and associated probabilities'.format(in_args.top_k))
    probs, classes = nn_model.predict(in_args.input_image, model, in_args.top_k, in_args.gpu)

    nn_model.print_predictions(probs, classes, in_args.category_names_file)
    

def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("input_image", type=str,
                help="Path to the input image file")
    
    parser.add_argument("checkpoint", type=str,
                help="Path to the saved model checkpoint")
    
    parser.add_argument("--top_k", type=int,
                help="Return top k most likely classes", default=1)

    parser.add_argument("--category_names_file", type=str, 
                help="Path to JSON file containing mapping of categories to real names", 
                default="cat_to_name.json")
    
    parser.add_argument("--gpu", action="store_true",
                dest='gpu', help="if passed, uses GPU for training", default=False)
    
    return parser.parse_args()
    
if __name__ == "__main__":
    main()
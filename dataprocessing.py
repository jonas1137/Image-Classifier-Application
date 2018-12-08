# PROGRAMMER: Jonas Persson
# DATE CREATED: 05/22/2018                                  
# REVISED DATE: -
# PURPOSE: This file contains all the functions related to data processing, such as data loading and image formatting.
# Functions:
#  load_data - load and process the image data to use for training
#  process_image - crop and normalize input image and return as tensor
#  assign_cat_to_name - create list of category names corresponding to category indices
#  load_checkpoint - load model and parameters from a saved checkpoint
#

import numpy as np
import json

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def load_data():
    """
    Load the data, define transforms and create dataloaders to be used as input for training the model.
     Returns:
      dataloaders = dataloader of images to be provided to the model with set batch size
      image_datasets = dataset of images with transforms applied
    """
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms for the training, validation, and testing sets
    data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224), #transforms.Resize(255),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(degrees=1),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                       'test': transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])]),
                       'valid': transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])}

    # Load the datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
                      'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])}

    # Define the dataloaders
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64),
                   'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32)}
    
    return dataloaders, image_datasets
        
def process_image(image):
    """
    Crop and normalize input image and return as tensor to be used for prediction using the model.
     Returns:
      tensor_image = tensor containing image data of numpy array
    """
    size = 224, 224
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    #Resize image proportionally setting the height to 256
    image = image.resize((int(256*image.size[0]/image.size[1]),256))
    
    new_width = 224
    new_height = 224
    
    #Calculate coordinates of the four corners of the center crop
    left_bound = int(image.size[0] - new_width)/2
    top_bound = int(image.size[1] - new_height)/2
    right_bound = int(image.size[0] + new_width)/2
    bottom_bound = int(image.size[1] + new_height)/2
    
    #Crop image accordin to above defined corner points
    pil_image = image.crop((left_bound, top_bound, right_bound, bottom_bound))

    #Convert to array of RGB-values
    np_image = np.array(pil_image)
    
    #Convert the rgb color channel values of the picture to (0-255) to values between 0-1 which is what the model expects
    np_image = np_image/255
    
    #Noramlize picture
    np_image = (np_image - mean) / std #np_image / std - mean #
    
    #Adapt the np_image to the format Pytorch expects by moving the color channel (in numpy and PIL = last dimension)
    #to the first dimension (Pytorch has color channel as first dimension)
    np_image = np_image.transpose((2, 0, 1))
    
    #Return as tensor
    tensor_image = torch.from_numpy(np_image).float()
    return tensor_image
                    
def assign_cat_to_name(filepath, categories):
    """
    Creates list of category names corresponding to the provided category indices.
     Returns:
      category_names - list of category names corresponding to the input category indices
    """
    #Retrieve category names
    with open(filepath, 'r') as f:
       cat_to_name = json.load(f)
    
    #Assign flower names corresponding to the category values using "cat_to_name.json" dictionary
    category_names = []
    for element in categories:
        category_names.append(cat_to_name.get(element))
            
    #Return vector of category names
    return category_names
        
def load_checkpoint(filepath, gpu):
    if gpu:
        checkpoint = torch.load(filepath)
    else:
        #When loading a checkpoint of a network trained on a GPU to be run on the CPU you need to remap the
        #tensor location using the map_location argument onto the CPU from the GPU otherwise you get an error.
        #The below solution to address this problem when loading the data was presented on the pytorch developer forums:
        checkpoint = torch.load(filepath, map_location = lambda storage, loc: storage)
    
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    epochs = checkpoint['epochs']
    class_to_idx = checkpoint['class_to_idx']
    
    return model, optimizer, epochs, class_to_idx
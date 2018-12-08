# PROGRAMMER: Jonas Persson
# DATE CREATED: 05/22/2018                                  
# REVISED DATE: -
# PURPOSE: Train an existing pytorch model downloaded from torchvision, and give
#          the user the option to define the hyperparameters of the model.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py --dir <directory to save checkpoints> --arch <model> --learning_rate <learning rate provided to optimizer>
#              --hidden_units <number of hidden units in the hidden layer(s)> --epochs <num of epochs>
#              --gpu <use cuda for training - default=False> --check_complex <model specific checkpoint filename - default=False>
#   Example call:
#    python train.py --dir 'checkpoints_folder/' --arch 'vgg' --gpu --check_complex

import argparse

from model import download_model, recreate_classifier, train_model, test_model
from dataprocessing import load_data

def main():
    in_arg = get_input_args()
    
    print("Command Line Arguments:\n dir =", in_arg.dir, "\n arch =", in_arg.arch, "\n learning_rate =", in_arg.learning_rate,
          "\n hidden_units =", in_arg.hidden_units, "\n epochs =", in_arg.epochs, "\n gpu =", in_arg.gpu,
          "\n check_complex =", in_arg.checkpoint_complex, "\n")
    
    #Download and set the chosen model
    model = download_model(in_arg.arch)
    
    #Load the input data
    dataloaders, datasets = load_data()
    
    #Save class_to_idx dictionary of dataset inside the model
    model.class_to_idx = datasets['train'].class_to_idx
    
    #Recreate the classifier for use with chosen model
    model = recreate_classifier(in_arg.arch, model, in_arg.hidden_units)
    print(in_arg.arch.upper(), "classifier configuration:\n", model.classifier, "\n")
    
    #Set the checkpoint directory and name
    if in_arg.checkpoint_complex:
        checkpoint_name = "{}checkpoint-{}-hiddenunits{}-lr{}-epochs{}.pth".format(in_arg.dir, in_arg.arch, in_arg.hidden_units,                                    in_arg.learning_rate, in_arg.epochs)
    else:
        checkpoint_name = "checkpoint.pth"
    
    #Train the model and save checkpoint
    model = train_model(model, dataloaders, in_arg.learning_rate, in_arg.epochs, in_arg.gpu, checkpoint_name)
    
    #Test the model
    test_model(model, dataloaders, in_arg.gpu)
    

def get_input_args():
    parser = argparse.ArgumentParser()

    # Argument 1: filepath
    parser.add_argument('--dir', type = str, default = '',
                        help = 'directory to save checkpoints')
    
    # Argument 2: architecture
    parser.add_argument('--arch', type = str, default = 'densenet',
                        help = 'chosen model')
    
    # Argument 3: learning rate
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                        help = 'learning rate')
    
    # Argument 4: hidden units
    parser.add_argument('--hidden_units', type = int, default = 512,
                        help = 'hidden units')
    
    # Argument 5: number of epochs
    parser.add_argument('--epochs', type = int, default = 3,
                        help = 'number of epochs to use in training')
    
    # Argument 6: use gpu
    parser.add_argument('--gpu', dest='gpu', action='store_true', default = False,
                        help = 'use gpu for training')
    
    # Argument 7: use model specific checkpoint file name = checkpoint file name that includes model parameters
    parser.add_argument('--checkpoint_complex', dest='checkpoint_complex', action='store_true', default = False,
                    help = 'use model specific checkpoint file name')

    return parser.parse_args() #in_args

main()
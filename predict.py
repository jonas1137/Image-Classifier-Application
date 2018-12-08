# PROGRAMMER: Jonas Persson
# DATE CREATED: 05/22/2018
# REVISED DATE: -
# PURPOSE: Predict the category of a flower picture using a saved pretrained model loaded
#          from a checkpoint. Display the predicted top K-values and corresponding probabilities.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py --dir <file path of image> --top_k <number of top predictions to display>
#.             --category_names <file path of json file containing dictionary of category names>
#              --gpu <use cuda for training - default=False> --checkpoint <checkpoint file path to load>
#   Example call:
#    python predict.py --dir 'flowers/valid/12/image_03997.jpg' --top_k 5 --gpu --checkpoint 'checkpoint-vgg-hiddenunits512-lr0.001-epochs3.pth'

import argparse

import torch
from torch.autograd import Variable

from model import predict_image

from dataprocessing import assign_cat_to_name,  load_checkpoint

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json

def main():
    in_arg = get_input_args()

    print("Command Line Arguments:\n image file path =", in_arg.dir, "\n top_k =", in_arg.top_k,
          "\n category_names =", in_arg.category_names, "\n gpu =", in_arg.gpu, "\n checkpoint file path =", in_arg.checkpoint)

    #Load model and corresponding attributes from checkpoint
    model, optimizer, epochs, class_to_idx = load_checkpoint(in_arg.checkpoint, in_arg.gpu)

    #Predict the category of the image (results vector of probabilities and matching categories)
    probabilities, categories = predict_image(in_arg.dir, model, in_arg.top_k, in_arg.gpu)

    #Assign names to the default category indices if a file is provided in the input args
    if in_arg.category_names:
        categories = assign_cat_to_name(in_arg.category_names, categories)

    #Print the most likely class/top K most likely classes
    if in_arg.top_k <= 1:
        print("The most likely category as predicted by the model is the", categories[0].upper(),
              "with a probability of", round(probabilities[0],3))
    else:
        print("The top K most likely categories and probabilies as predicted by the model are:")
        for cat, prob in zip(categories, probabilities):
            print("%20s: %4.3f" % (cat.upper(), prob))




    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    #Assign flower names corresponding to the category values using "cat_to_name" function
    #flower_names = []
    #for element in categories:
    #    flower_names.append(cat_to_name.get(element))

    #Open image
    image = Image.open(in_arg.dir)

    #Plot the flower picture and probability diagram
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,10))
    ax1.set_title(categories[0])
    ax1.imshow(image, aspect="auto")
    ax1.axis('off')

    y_order = np.arange(len(categories))
    ax2.barh(y_order, probabilities, align = 'center', color = 'blue', ecolor = 'black')
    ax2.set_yticks(y_order)
    ax2.set_yticklabels(categories)
    ax2.invert_yaxis()

    #plt.tight_layout()

    f.savefig('test.png',bbox_inches='tight')



def get_input_args():
    parser = argparse.ArgumentParser()

    # Argument 1: image filepath
    parser.add_argument('--dir', type = str, default = 'flowers/test/10/image_07090.jpg',
                    help = 'image file path')

    # Argument 2: top K values
    parser.add_argument('--top_k', type = int, default = 1,
                    help = 'number of most likely categories to show')

    # Argument 3: file containing dictionary of category names
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                    help = 'file path of category names')

    # Argument 4: use gpu
    parser.add_argument('--gpu', dest='gpu', action='store_true', default = False,
                    help = 'use gpu for training')

    # Argument 5: checkpoint file path to load
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth',
                    help = 'checkpoint file path')

    return parser.parse_args()

main()

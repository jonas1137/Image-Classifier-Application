# PROGRAMMER: Jonas Persson
# DATE CREATED: 05/22/2018
# REVISED DATE: -
# PURPOSE: This file contains all the functions related to the model, such as training and test functions.
# Functions:
#  download_model - download the model
#  recreate_classifier - recreate the classifier of the supplied model
#  train_model - train the model
#  test_model - test the model
#  predict_image - predict category of supplied image using the trained model
#

from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from dataprocessing import process_image

def download_model(model_name):
    #model = models.model(pretrained=True) #densenet121
    if model_name.startswith('alexnet'):
        model = models.alexnet(pretrained=True)
    elif model_name.startswith('vgg'):
        model = models.vgg11_bn(pretrained=True)
    elif model_name.startswith('densenet'):
        model = models.densenet121(pretrained=True)

    #Lock the features part of the model
    for param in model.parameters():
        param.requires_grad = False

    return model

def recreate_classifier(model_name, model, hidden_units):
    """
    Recreates the classifier of three different torchvision models: alexnet, vgg and densenet.
     Returns:
      model - the input model is returned, updated with the new classifier
    """

    from collections import OrderedDict

    if model_name.startswith('alexnet'):
        classifier = nn.Sequential(OrderedDict([
                                  ('dropout1', nn.Dropout(0.5)),
                                  ('fc1', nn.Linear(9216, hidden_units, bias=True)),
                                  ('relu1', nn.ReLU()),
                                  ('dropout2', nn.Dropout(0.5)),
                                  ('fc2', nn.Linear(hidden_units, hidden_units, bias=True)),
                                  ('relu2', nn.ReLU()),
                                  ('fc3', nn.Linear(hidden_units, 102, bias=True)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

    if model_name.startswith('vgg'):
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, hidden_units, bias=True)),
                                  ('relu1', nn.ReLU()),
                                  ('dropout1', nn.Dropout(0.5)),
                                  ('fc2', nn.Linear(hidden_units, hidden_units, bias=True)),
                                  ('relu2', nn.ReLU()),
                                  ('dropout2', nn.Dropout(0.5)),
                                  ('fc3', nn.Linear(hidden_units, 102, bias=True)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

    if model_name.startswith('densenet'):
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, hidden_units)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(0.5)),
                                  ('fc2', nn.Linear(hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

    model.classifier = classifier
    return model

def train_model(model, dataloaders, learning_rate, epochs, gpu, checkpoint_name):
    """
    Train the model using the input dataloaders and hyperparameters: learning rate and epochs.
    A checkpoint is saved at end of training.
     Returns:
      model - the trained model is returned
    """
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    steps = 0
    print_every = 20

    print("cuda", torch.cuda.is_available())
    if gpu and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    print("Training model...:")
    for e in range(epochs):
        #Training mode, turn off dropout
        model.train()

        running_loss = 0

        for ii, (images, labels) in enumerate(dataloaders['train']):
            steps += 1
            print(steps)
            images, labels = Variable(images), Variable(labels)
            optimizer.zero_grad()

            if gpu and torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if steps % print_every == 0:
                #Enter eval mode for validation = turn off dropout
                model.eval()

                valid_loss = 0
                valid_accuracy = 0

                for ii, (images, labels) in enumerate(dataloaders['valid']):

                    images, labels = Variable(images, volatile = True), Variable(labels, volatile = True)

                    if gpu and torch.cuda.is_available():
                        images, labels = images.cuda(), labels.cuda()

                    output = model.forward(images)
                    valid_loss += criterion(output, labels)

                    #Get the probabilities by taking exponential of output which is in log-softmax form
                    ps = torch.exp(output).data

                    #Test the matrix of classified images (batchsize x number of categories = 32x102)
                    #for correctness, by comparing the category with the greatest value of each row
                    #with the corresponding label. If they are equal the number 1 is assigned otherwise 0
                    #(each row corresponds to one image = 102 columns with the predicted probability of
                    #the image belonging to that particular category)

                    equality = (labels.data == ps.max(1)[1])

                    #Get accuracy by taking mean of all predictions
                    valid_accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Training Loss: {:.3f}".format(running_loss/print_every), #{:.3f}
                      "Validation Loss: {:.4f}".format(float(valid_loss/len(dataloaders['valid']))),
                      "Validation Accuracy: {:.3f}".format(valid_accuracy/len(dataloaders['valid'])))

                #Reset the running loss b
                running_loss = 0

                #Turn dropout back on when training continues
                model.train()

    #Save checkpoint as dictionary of model parameters
    checkpoint = {'model': model,
                  'optimizer': optimizer,
                  'optimizer_dict': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'epochs': epochs + 1,
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, checkpoint_name)
    print("Checkpoint saved!")

    return model


def test_model(model, dataloaders, gpu):
    """
    Test the trained model to verify that it is accurate on new set of images not used in training.
     Returns:
      -
    """
    criterion = nn.NLLLoss()

    if gpu and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    #Set to eval which switches off dropout during evaluation of the model
    model.eval()

    test_loss = 0
    test_accuracy = 0

    for ii, (inputs, labels) in enumerate(dataloaders['test']):

        if gpu and torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).data[0]

        #Get the probabilities by taking exponential of output which is in log-softmax form
        ps = torch.exp(output).data

        #Test the matrix of classified images (batchsize x number of categories = 32x102)
        #for correctness, by comparing the category with the greatest value of each row
        #with the corresponding label. If they are equal the number 1 is assigned otherwise 0
        #(each row corresponds to one image = 102 columns with the predicted probability of
        #the image belonging to that particular category)
        equality = (labels.data == ps.max(1)[1])

        #Get accuracy by taking mean of all predictions
        test_accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Test Loss: {:.4f}".format(test_loss/len(dataloaders['test'])),
          "Test Accuracy: {:.3f}".format(test_accuracy/len(dataloaders['test'])))


def predict_image(image_path, model, topk, gpu):
    """
    Predict the category of the input image using the input model. Return top K most likely categories.
     Returns:
      probabilities - List of probabilities for the top K most likely categories
      category_indices - List of category indices for the top K most likely categories
    """
    #Set model to eval mode to turn off dropout
    model.eval()

    if gpu and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    img = Image.open(image_path)
    img = process_image(img)
    img_clone = img.clone()
    img = img_clone.resize_(1,3,224,224)
    img = Variable(img, volatile=True)

    if gpu and torch.cuda.is_available():
        img = img.cuda()
    else:
        img = img.cpu()

    output = model.forward(img)

    #Calculate the probabilites vector
    ps = torch.exp(output)

    #Get the tensors of top K moste likely classes and corresponding probabilities
    probabilities, indices = ps.topk(topk)

    #Error message: "Can't convert CUDA tensor to numpy (it doesn't support GPU arrays). Use .cpu() to move the tensor to host memory first", hence they are moved to CPU on next line:
    probabilities, indices = probabilities.cpu(), indices.cpu()

    #Retrieve the data from the tensor
    probabilities = probabilities.data.numpy()[0]
    indices = indices.data.numpy()[0]

    #Convert numpy array to list
    probabilities = probabilities.tolist()
    indices = indices.tolist()

    #Invert the class_to_idx list to put the class in the value field
    class_to_idx_reverse = dict((v,k) for k,v in model.class_to_idx.items())

    #Create list of true category indices by retrieving them from class_to_idx_reverse using the indices of the model as keys
    category_indices = []
    for element in indices:
        category_indices.append(class_to_idx_reverse.get(element))

    #Return the vector of probabilities and matching categories
    return(probabilities, category_indices)

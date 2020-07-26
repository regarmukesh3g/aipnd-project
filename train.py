import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import argparse
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import time


print(torch.cuda.is_available())

def get_parser():
    """
    Cretes Argument parser
    :return:
    """
    parser = argparse.ArgumentParser(description='Trains the data model')
    parser.add_argument('training_data_path', metavar='Data_path', type=str,
                        help='path of the dataset')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='directory path for saving model')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for the model')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='hidden units in model classifier')
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of epochs in training')
    parser.add_argument('--arch', type=str, default='densenet121',
                        help='type of architecture you want to use')
    parser.add_argument('--gpu', action='store_const', const=True,
                        help='add if need gpu')
    return parser


def get_loader(dir):
    """
    Return Load Paths according to directory
    :param dir: Directory Path
    :return: Data loader
    """
    dir = dir

    train_transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.RandomRotation(30),
                                          transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5],
                                                         [0.5, 0.5, 0.5])])
    data = datasets.ImageFolder(dir, transform=train_transform)

    loader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True)
    return loader


def get_model(model_name):
    if model_name == "vgg19":
        model = models.vgg19(pretrained=True)
    if model_name == "densenet121":
        model = models.densenet121(pretrained=True)

    return model

def get_device(gpu= False):

    if gpu:
        return 'cuda'
    else:
        return 'cpu'


def train(model, train_loader, epochs ,device, learning_rate):
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)
    # loss function
    criterion = nn.NLLLoss()
    model.to(device)
    torch.cuda.empty_cache()
    for epoch in range(epochs):
        steps = 0
        print_every = 25
        running_loss = 0
        #print(enumerate(train_loader))
        print("Epoch = ",epoch)
        for ii, (inputs, labels) in enumerate(train_loader):

            #print(data)
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            # optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

            _, predict = (outputs.max, 1)
            del inputs, labels
            if steps % print_every == 0:
                # print(outputs)
                # print(labels)
                # print(running_loss)
                print("Epoch: {}/{}... ".format(epoch + 1, epochs),
                      "Loss: {:.4f}".format(running_loss / print_every))

                running_loss = 0
    return model
def validate(model, validate_loader, device):
    criterion = nn.NLLLoss()
    print_every = 2
    running_loss = 0
    total = 0
    correct = 0
    steps = 0
    # print(enumerate(train_loader))
    for ii, (inputs, labels) in enumerate(validate_loader):
        torch.cuda.empty_cache()
        # print(data)
        steps += 1

        inputs, labels = inputs.to(device), labels.to(device)

        # optimizer.zero_grad()

        # Forward and backward passes
        outputs = model.forward(inputs)
        # print(outputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del inputs, labels
        if steps % print_every == 0:
            # print(outputs)
            # print(labels)
            # print(running_loss)
            print( "Loss: {:.4f}".format(running_loss / print_every),
                  "Accuracy: {:.4f}".format(100.0 * correct/total))
            correct = 0
            total = 0

            running_loss = 0




def main():

    parser = get_parser()
    args = parser.parse_args()
    print(args.training_data_path)
    train_loader = get_loader(args.training_data_path + '/train')

    #print(train_loader)
    num_of_classes = len(train_loader.dataset.classes)
    #print(args)
    hidden_units = args.hidden_units
    model = get_model(args.arch)
    model_classifier_in = 0
    if model.classifier.__class__ == nn.Sequential:
        model_classifier_in = model.classifier[0].in_features
    else:
        model_classifier_in =  model.classifier.in_features
    print(model_classifier_in)
    classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(model_classifier_in, hidden_units)),
                                            ('relu1', nn.ReLU()),
                                            ('fc2', nn.Linear(hidden_units, num_of_classes)),
                                            ('out',nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    # classfier_in_features = model.classifier[-1].out_features
    #model.classifier[-1].out_features = num_of_classes

    #print(classfier_in_features)

    learning_rate =  args.learning_rate
    device = get_device(args.gpu)
    epochs = args.epochs
    train(model, train_loader, epochs, device, learning_rate)
    c = train_loader.dataset.class_to_idx
    data_dict = {
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'mapping': c,
        'classfier_in': model_classifier_in,
        'hidden_units' : hidden_units,
        'output': num_of_classes,
        'arch': args.arch
    }
    torch.save(data_dict, args.save_dir + '/checkpoint.pth')
    validate_loader = get_loader(args.training_data_path + '/valid')
    validate(model, validate_loader, device)


    del model



if __name__ == "__main__":
    main()


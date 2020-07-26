

import torch
import argparse
import json
from torch import nn
from collections import OrderedDict
from PIL import Image
from torchvision import models, transforms
import numpy as np

def get_parser():
    """
    Cretes Argument parser
    :return:
    """
    parser = argparse.ArgumentParser(description='Trains the data model')
    parser.add_argument('input_image', metavar='Image_Path', type=str,
                        help='path of the image')
    parser.add_argument('checkpoint', metavar='Checkpoint', type=str,
                        help='path of the model checkpoint')
    parser.add_argument('--top_k', type=int, default=4,
                        help='number of top categories')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='file path for category json')
    parser.add_argument('--gpu', action='store_const', const=True,
                        help='add if need gpu')
    return parser

def load_model(checkpoint_path):
    model_dict = torch.load(checkpoint_path)

    #print(model_dict)

    classifier_in = model_dict['classfier_in']
    hidden_units = model_dict['hidden_units']
    out_num = model_dict['output']
    model_name = model_dict['arch']
    model = None
    if model_name == 'densenet121':
        model = models.densenet121()
    if model_name == 'vgg19':
        model = models.vgg19()


    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_in, hidden_units)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, out_num)),
        ('out', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    model.load_state_dict(model_dict['state_dict'])


    return model

def process_image(image_path, device):
    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                               [0.5, 0.5, 0.5])])
    image = valid_transform(Image.open(image_path))

    image = image.to(device)
    image = image.view(1, 3, 224, 224)
    return image

def predict(model , image_path, device):

    image = process_image(image_path,device)
    output = model(image)

    predicted = output.topk(5)
    print(predicted)
    return predicted[1].detach().numpy()[0], predicted[0].detach().numpy()[0]

def get_device(gpu= False):

    if gpu:
        return 'cuda'
    else:
        return 'cpu'

def main():
    parser = get_parser()
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    model = load_model(checkpoint_path)
    image_path = args.input_image
    device = get_device(args.gpu)
    output, probs = predict(model, image_path, device=device)
    #print(str(output))
    cat_to_name = {}
    if args.category_names:
        with open(args.category_names,'r') as cat_file:
            cat_to_name = json.load(cat_file)
    names_list = []
    for i in output:
        names_list.append(cat_to_name.get(str(i), str(i)))

    i = 0
    for name in names_list:
        print(name, np.exp(probs[i]))
        i = i + 1

if __name__ == '__main__':
    main()

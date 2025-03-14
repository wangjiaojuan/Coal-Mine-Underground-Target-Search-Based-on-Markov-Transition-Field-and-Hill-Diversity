import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
import numpy as np

from ShuffleNet import shufflenet_v2_x0_5
from Resnet import resnet18
from Vgg import vgg16

import warnings
warnings.filterwarnings("ignore", category=Warning)

def get_feature_shuffleNetV2(image_dir):#ok
    shuffleNetV2_model = shufflenet_v2_x0_5(pretrained=True)
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    im = Image.open(image_dir).convert('RGB')
    im = trans(im)
    im.unsqueeze_(dim=0)
    shuffleNetV2_model = shuffleNetV2_model.eval()
    features,_ = shuffleNetV2_model(im)
    feature = features.data.cpu().numpy()
    feature = feature.tolist()
    feature  = feature[0]
    return feature
def get_feature_resnet18(image_dir):#ok
    resnet_model = resnet18(pretrained=True)
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    im = Image.open(image_dir).convert('RGB')
    im = trans(im)
    im.unsqueeze_(dim=0)
    resnet_model = resnet_model.eval()
    features,_ = resnet_model(im)
    feature = features.data.cpu().numpy()
    feature = feature.tolist()
    feature  = feature[0]
    return feature
def get_feature_vgg16(image_dir):#ok
    vgg_model = vgg16(pretrained=True)
    new_classifier = torch.nn.Sequential(*list(vgg_model.children())[-1][:6])
    vgg_model.classifier = new_classifier
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    im = Image.open(image_dir).convert('RGB')
    im = trans(im)
    im.unsqueeze_(dim=0)
    vgg_model = vgg_model.eval()
    y = vgg_model(im).data.numpy().tolist()
    feature  = y[0]
    return feature
def get_feature_densenet121(image_dir):#ok
    densenet_model121 = models.densenet121(pretrained=True)
    features_model=list(densenet_model121.children())[:-1]
    densenet_model=torch.nn.Sequential(*features_model)
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    im = Image.open(image_dir).convert('RGB')
    im = trans(im)
    im.unsqueeze_(dim=0)
    densenet_model = densenet_model.eval()
    features = densenet_model(im)
    feature = features.data.cpu().numpy()
    feature = feature.flatten()
    feature = feature.tolist()
    return feature
def get_feature_Alexnet(image_dir):#ok
    from torchvision.models.alexnet import AlexNet_Weights
    Alexnet_model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    Alexnet_model.classifier = Alexnet_model.classifier[:-1]
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    im = Image.open(image_dir).convert('RGB')
    im = trans(im)
    im.unsqueeze_(dim=0)
    Alexnet_model = Alexnet_model.eval()
    features = Alexnet_model(im)
    feature = features.data.cpu().numpy()
    feature = feature.flatten()
    feature = feature.tolist()
    return feature
def get_feature_squeezenet(image_dir):#ok
    squeezenet_model =models.squeezenet1_0(pretrained=True)
    #squeezenet_model.classifier = squeezenet_model.classifier[:-1]
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    im = Image.open(image_dir).convert('RGB')
    im = trans(im)
    im.unsqueeze_(dim=0)
    squeezenet_model = squeezenet_model.eval()
    features = squeezenet_model(im)
    feature = features.data.cpu().numpy()
    feature = feature.flatten()
    feature = feature.tolist()
    return feature
def get_feature_convnext(image_dir):#ok
    ConvNeXt_model =models.convnext_base(ConvNeXt=True, pretrained=True)
    ConvNeXt_model.classifier = ConvNeXt_model.classifier[:-1]
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    im = Image.open(image_dir).convert('RGB')
    im = trans(im)
    im.unsqueeze_(dim=0)
    ConvNeXt_model = ConvNeXt_model.eval()
    features = ConvNeXt_model(im)
    feature = features.data.cpu().numpy()
    feature = feature.flatten()
    feature = feature.tolist()
    return feature
def get_feature_efficientnet(image_dir):#ok
    efficientnet_model =models.efficientnet_v2_s(weights = models.efficientnet.EfficientNet_V2_S_Weights)
    efficientnet_model.classifier = efficientnet_model.classifier[:-1]
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    im = Image.open(image_dir).convert('RGB')
    im = trans(im)
    im.unsqueeze_(dim=0)
    efficientnet_model = efficientnet_model.eval()
    features = efficientnet_model(im)
    feature = features.data.cpu().numpy()
    feature = feature.flatten()
    feature = feature.tolist()
    return feature
def get_feature_googlenet(image_dir):#ok
    googlenet_model =models.googlenet(pretrained=True)
    features_model=list(googlenet_model.children())[:-1]
    googlenet_model=torch.nn.Sequential(*features_model)
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    im = Image.open(image_dir).convert('RGB')
    im = trans(im)
    im.unsqueeze_(dim=0)
    googlenet_model = googlenet_model.eval()
    features = googlenet_model(im)
    feature = features.data.cpu().numpy()
    feature = feature.flatten()
    feature = feature.tolist()
    return feature
def get_feature_mnasnet(image_dir):  #ok
    mnasnet_model =models.mnasnet0_5(weights = models.mnasnet.MNASNet0_5_Weights)
    mnasnet_model.classifier = mnasnet_model.classifier[:-1]
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    im = Image.open(image_dir).convert('RGB')
    im = trans(im)
    im.unsqueeze_(dim=0)
    mnasnet_model = mnasnet_model.eval()
    features = mnasnet_model(im)
    feature = features.data.cpu().numpy()
    feature = feature.flatten()
    feature = feature.tolist()
    return feature
def get_feature_mobilenet(image_dir):#ok
    mobilenet_model =models.mobilenet.mobilenet_v2(pretrained=True) 
    mobilenet_model.classifier = mobilenet_model.classifier[:-1]
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    im = Image.open(image_dir).convert('RGB')
    im = trans(im)
    im.unsqueeze_(dim=0)
    mobilenet_model = mobilenet_model.eval()
    features = mobilenet_model(im)
    feature = features.data.cpu().numpy()
    feature = feature.flatten()
    feature = feature.tolist()
    return feature
def get_feature_regnet(image_dir):#ok
    regnet_model =models.regnet.regnet_x_1_6gf(weights = models.regnet.RegNet_X_1_6GF_Weights)
    features_model=list(regnet_model.children())[:-1]
    regnet_model=torch.nn.Sequential(*features_model)
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    im = Image.open(image_dir).convert('RGB')
    im = trans(im)
    im.unsqueeze_(dim=0)
    regnet_model = regnet_model.eval()
    features = regnet_model(im)
    feature = features.data.cpu().numpy()
    feature = feature.flatten()
    feature = feature.tolist()
    return feature

def get_img_feature_shuffleNetV2_model(img_dir):
    img_feature = get_feature_shuffleNetV2(img_dir)
    return img_feature
def get_img_feature_vgg16(img_dir):
    img_feature = get_feature_vgg16(img_dir)
    return img_feature
def get_img_feature_resnet18(img_dir):
    img_feature = get_feature_resnet18(img_dir)
    return img_feature
def get_img_feature_densenet(img_dir):
    img_feature = get_feature_densenet121(img_dir)
    return img_feature
def get_img_feature_Alexnet(img_dir):
    img_feature = get_feature_Alexnet(img_dir)
    return img_feature
def get_img_feature_squeezenet(img_dir):
    img_feature = get_feature_squeezenet(img_dir)
    return img_feature
def get_img_feature_convnext(img_dir):
    img_feature = get_feature_convnext(img_dir)
    return img_feature
def get_img_feature_efficientnet(img_dir):
    img_feature = get_feature_efficientnet(img_dir)
    return img_feature
def get_img_feature_googlenet(img_dir):
    img_feature = get_feature_googlenet(img_dir)
    return img_feature
def get_img_feature_mnasnet(img_dir):
    img_feature = get_feature_mnasnet(img_dir)
    return img_feature
def get_img_feature_mobilenet(img_dir):
    img_feature = get_feature_mobilenet(img_dir)
    return img_feature
def get_img_feature_regnet(img_dir):
    img_feature = get_feature_regnet(img_dir)
    return img_feature


def get_Datasets_feature_preAnalytical(filepath):
    file = open(filepath)
    filelist = file.readlines()
    file.close()    

    all_feature_vgg16 = []
    all_feature_resnet18 = []
    all_feature_shuffleNetV2 = []
    all_feature_densenet121 = []
    all_feature_Alexnet = []
    all_feature_squeezenet = []
    all_feature_convnext = []
    all_feature_efficientnet = []
    all_feature_googlenet = []
    all_feature_mnasnet = []
    all_feature_mobilenet = []
    all_feature_regnet = []

    num=0
    for fi in filelist:
        num=num+1
        print(num,len(filelist))
        img_dir = fi[:-1]

        all_feature_shuffleNetV2.append(get_img_feature_shuffleNetV2_model(img_dir))
        all_feature_vgg16.append(get_img_feature_vgg16(img_dir))
        all_feature_resnet18.append(get_img_feature_resnet18(img_dir))
        all_feature_densenet121.append(get_img_feature_densenet(img_dir))
        all_feature_Alexnet.append(get_img_feature_Alexnet(img_dir))
        all_feature_squeezenet.append(get_img_feature_squeezenet(img_dir))
        all_feature_convnext.append(get_img_feature_convnext(img_dir))
        all_feature_googlenet.append(get_img_feature_googlenet(img_dir))
        all_feature_efficientnet.append(get_img_feature_efficientnet(img_dir))
        all_feature_mnasnet.append(get_img_feature_mnasnet(img_dir))
        all_feature_regnet.append(get_img_feature_regnet(img_dir))
        all_feature_mobilenet.append(get_img_feature_mobilenet(img_dir))

        Result = [all_feature_shuffleNetV2,all_feature_vgg16,all_feature_resnet18,all_feature_densenet121,
                  all_feature_Alexnet,all_feature_convnext,all_feature_squeezenet,
                  all_feature_googlenet,all_feature_efficientnet,all_feature_mnasnet,
                  all_feature_regnet,all_feature_mobilenet]
    return Result

def get_Datasets_feature_afterAnalytical(filepath):
    file = open(filepath)
    filelist = file.readlines()
    file.close()    

    all_feature_vgg16 = []
    all_feature_resnet18 = []
    all_feature_shuffleNetV2 = []
    all_feature_densenet121 = []
    all_feature_Alexnet = []
    all_feature_squeezenet = []
    all_feature_convnext = []
    all_feature_efficientnet = []
    all_feature_googlenet = []
    all_feature_mnasnet = []
    all_feature_mobilenet = []
    all_feature_regnet = []

    num=0
    for fi in filelist:
        num=num+1
        print(num,len(filelist))
        img_dir = fi[:-1]

        high_freq_image_dir = img_dir.replace('Datasets','high_freq_image')
        markov_image_dir = img_dir.replace('Datasets','markov_image')

        all_feature_shuffleNetV2.append(get_img_feature_shuffleNetV2_model(high_freq_image_dir) + get_img_feature_shuffleNetV2_model(markov_image_dir))
        
        
        all_feature_vgg16.append(get_img_feature_vgg16(high_freq_image_dir)+get_img_feature_vgg16(markov_image_dir))
        
        all_feature_resnet18.append(get_img_feature_resnet18(high_freq_image_dir)+get_img_feature_resnet18(markov_image_dir))
        
        all_feature_densenet121.append(get_img_feature_densenet(high_freq_image_dir)+get_img_feature_densenet(markov_image_dir))
        
        all_feature_Alexnet.append(get_img_feature_Alexnet(high_freq_image_dir)+get_img_feature_Alexnet(markov_image_dir))
        all_feature_squeezenet.append(get_img_feature_squeezenet(high_freq_image_dir)+get_img_feature_squeezenet(markov_image_dir))
        all_feature_convnext.append(get_img_feature_convnext(high_freq_image_dir)+get_img_feature_convnext(markov_image_dir))
        all_feature_googlenet.append(get_img_feature_googlenet(high_freq_image_dir)+get_img_feature_googlenet(markov_image_dir))
        all_feature_efficientnet.append(get_img_feature_efficientnet(high_freq_image_dir)+get_img_feature_efficientnet(markov_image_dir))
        all_feature_mnasnet.append(get_img_feature_mnasnet(high_freq_image_dir)+get_img_feature_mnasnet(markov_image_dir))
        all_feature_regnet.append(get_img_feature_regnet(high_freq_image_dir)+get_img_feature_regnet(markov_image_dir))
        all_feature_mobilenet.append(get_img_feature_mobilenet(high_freq_image_dir)+get_img_feature_mobilenet(markov_image_dir))

        Result = [all_feature_shuffleNetV2,all_feature_vgg16,all_feature_resnet18,all_feature_densenet121,
                  all_feature_Alexnet,all_feature_convnext,all_feature_squeezenet,
                  all_feature_googlenet,all_feature_efficientnet,all_feature_mnasnet,
                  all_feature_regnet,all_feature_mobilenet]
    return Result
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models
from torchsummary import summary

class Model:
    def __init__(self, model_type="vgg"):
        """
        Class to select and prepare a pre-trained model for feature extraction.
        
        Args:
        - model_type (str): Type of pre-trained model to use (default is "vgg").
        """
        self.model_type = model_type  # Assign model type to the class attribute
        self.model = self.model_select()  # Select the pre-trained model based on the model type

    def model_select(self):
        """
        Select the pre-trained model based on the specified model type.
        
        Returns:
        - model (torch.nn.Module): Pre-trained model.
        """
        if self.model_type == "vgg":
            model = models.vgg16(pretrained=True)  # Load pre-trained VGG16 model
        elif self.model_type == "res":
            model = models.resnet18(pretrained=True)  # Load pre-trained ResNet18 model
        
        return model  # Return the selected pre-trained model

    def feature_extr_prep(self):
        """
        Prepare the feature extractor for the selected pre-trained model.
        
        Returns:
        - feature_extr (torch.nn.Module): Feature extractor module.
        """
        if self.model_type == "vgg":
            feature_extr = torch.nn.Sequential(*list(self.model.features.children()))  # Extract features from VGG16 model
        elif self.model_type == "res":
            feature_extr = torch.nn.Sequential(*list(self.model.children())[:-1])  # Extract features from ResNet18 model

        # Freeze the weights of the pre-trained layers
        for param in feature_extr.parameters():
            param.requires_grad = False
        
        return feature_extr  # Return the prepared feature extractor


        
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms
from torchsummary import summary


# Create a custom Dataset class to apply the transformation batch-wise
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        """
        Custom dataset class to apply transformations to images batch-wise.
        
        Args:
        - data (np.ndarray): Numpy array containing image data.
        - transform (callable, optional): Optional transformation to be applied to the images.
        """

        self.data = data
        self.transform = transform

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Returns a sample from the dataset at the given index."""
        sample = self.data[index] # Get the sample at the specified index
        if self.transform: 
            sample = self.transform(sample) # Apply the transformation to the sample
        return sample


class Data_Loader_Preparator:
    """
        Class to prepare data loader for a folder containing image files.
        
        Args:
        - folder (str): Path to the folder containing image files.
        - batch_size (int, optional): Batch size for the data loader.
    """
    def __init__(self,folder, batch_size = 64):
        
        self.folder = folder
        self.batch_size = batch_size
    
    def transform_def(self):
        """
        Define the transformation to be applied to each image.
        
        Returns:
        - transform (torchvision.transforms.Compose): Composition of transformations.
        """

        # Define the transformation to be applied to each image
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert to PIL Image
            transforms.Resize((224, 224)), # Resize the image to (224, 224)
            transforms.ToTensor(), # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])
        
        return transform
    
    def load_dataset(self):
        """
        Load images from the folder and convert them to a numpy array.
        
        Returns:
        - images (np.ndarray): Numpy array containing image data.
        """
        files = sorted(os.listdir(self.folder))  # Get list of files in the folder and sort them
        images = [] # Initialize empty list to store image data
        
        # Initialize tqdm progress bar
        progress_bar = tqdm(total=len(files), desc='Loading Images', unit='image')

        for f in files: # Iterate through each file in the folder
            image_path = os.path.join(self.folder, f) # Construct path to the image file
            with open(image_path, 'rb') as file:
                img = Image.open(file) # Open the image file using PIL
                img_array = np.array(img)  # Convert PIL image to NumPy array
                images.append(img_array) # Append the image data to the list
            # Update progress bar
            progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        images = np.asarray(images)

        return images

    def Data_Loader_def(self):
        """
        Create a data loader for the images in the folder.
        
        Returns:
        - dataloader (torch.utils.data.DataLoader): DataLoader object for the dataset.
        """
        
        transform = self.transform_def() # Define the transformation
        image_dataset = self.load_dataset() # Load the dataset from the folder

        # Create a dataset with the original numpy array
        dataset = CustomDataset(image_dataset, transform=transform)

        # Create a data loader with the dataset
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=SequentialSampler(dataset))

        return dataloader 



class Img_load:
    def __init__(self,img):
        """
        Class to load and preprocess a single image.
        
        Args:
        - img (np.ndarray): Numpy array containing image data.
        """
        self.img = img
    
    def transform_def(self): 
        """
        Define the transformation to be applied to the image.
        
        Returns:
        - transform (torchvision.transforms.Compose): Composition of transformations.
        """
        # Define the transformation to be applied to each image
        transform = transforms.Compose([
            #transforms.ToPILImage(),  # Convert to PIL Image
            transforms.Resize((224, 224)), # Resize the image to (224, 224)
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])
        
        return transform
    
    def get_img(self):
        """
        Preprocess the image using the defined transformation.
        
        Returns:
        - img (torch.Tensor): Preprocessed image as a PyTorch tensor.
        """
        transform = self.transform_def() # Define the transformation

        img = transform(self.img) # Apply the transformation to the image

        return img


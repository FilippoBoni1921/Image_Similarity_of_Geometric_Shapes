import numpy as np
import os
import re
from PIL import Image
from tqdm import tqdm
import torch
import sys

from .data_prep import Data_Loader_Preparator,Img_load
from .models import Model

def dataset_feature_extr(data_folder,model_type, batch_size, device):
    """
    Extract features from images in the dataset folder.
    
    Args:
    - data_folder (str): Path to the folder containing the dataset images.
    - model_type (str): Type of model being used for feature extraction.
    - batch_size (int): Batch size for processing images.
    - device (torch.device): Device to perform computations (e.g., CPU or GPU).
    """
    # Print a message indicating feature extraction process has started
    print("Extracting the Features of the Dataset images")

    # Initialize the model for feature extraction
    model = Model(model_type)

    # Prepare data loader for dataset images
    data_load = Data_Loader_Preparator(data_folder, batch_size)

    # Initialize feature extractor
    feature_extractor = model.feature_extr_prep()

    # Get data loader
    dataloader =  data_load.Data_Loader_def()

    # Move feature extractor to specified device (e.g., GPU)
    feature_extractor.to(device)

    # Set the model to evaluation mode
    feature_extractor.eval()

    # List to store the features
    features_list = []

    # Initialize tqdm with the total number of batches
    progress_bar = tqdm(total=len(dataloader), desc="Feature Extract")

    # Process images in batches and extract features
    with torch.no_grad():
        for batch_images in dataloader:
            batch_images = batch_images.to(device)
            # Compute features for the batch of images
            batch_features = feature_extractor(batch_images)
            # Append the features to the features list
            features_list.append(batch_features.cpu().numpy())
            # Update the progress bar
            progress_bar.update(1)

    # Concatenate the features from all batches
    features = np.concatenate(features_list, axis=0)

    # Save the features to a file
    np.save("features.npy", features)


def img_feature_extr(img,model_type,device):
    """
    Extract features from a single image.
    
    Args:
    - img (PIL.Image): Image object from which features are to be extracted.
    - model_type (str): Type of model being used for feature extraction.
    - device (torch.device): Device to perform computations (e.g., CPU or GPU).
    
    Returns:
    - features (np.ndarray): Extracted features from the image.
    """
    # Initialize the model for feature extraction
    model = Model(model_type)

    # Initialize feature extractor
    feature_extractor = model.feature_extr_prep()  

    # Load the image
    img_loader = Img_load(img)
    img = img_loader.get_img()

    # Adjust image dimensions if required by the model
    if model_type == "res":
        img = img.unsqueeze(0)

    # Move feature extractor to specified device (e.g., GPU)
    feature_extractor.to(device)
    img = img.to(device)

    # Extract features from the image
    features = feature_extractor(img)

    # Convert features to numpy array
    features = features.cpu().numpy()

    return features


def cosine_similarity_np(array1, array2):
    """
    Compute cosine similarity between two numpy arrays.
    
    Args:
    - array1 (np.ndarray): First array.
    - array2 (np.ndarray): Second array.
    
    Returns:
    - cos_sim (np.ndarray): Cosine similarity between the two arrays.
    """
    # Flatten the arrays
    array1_flat = array1.reshape(1, -1)
    array2_flat = array2.reshape(array2.shape[0], -1)

    # Compute dot product
    dot_product = np.dot(array1_flat, array2_flat.T)

    # Compute magnitudes
    magnitude_array1 = np.sqrt(np.sum(array1_flat**2))
    magnitude_array2 = np.sqrt(np.sum(array2_flat**2, axis=1))

    # Compute cosine similarity
    cos_sim = dot_product / (magnitude_array1 * magnitude_array2)

    return cos_sim


def geom_fam_mask_gen(file,folder):
    """
    Generate mask indices for images belonging to the same geometric family as the input file.
    
    Args:
    - file (str): Name of the input file.
    - folder (str): Path to the folder containing the images.
    
    Returns:
    - geom_fam_indices (list): List of indices of images belonging to the same geometric family.
    """
    # Extract the first part of the file name
    first_part = re.match(r"([^_]+)", file).group(1)

    # Get the list of files in the folder
    files = sorted(os.listdir(folder))

    # Filter indices of images belonging to the same geometric family
    geom_fam_indices = [index for index, file in enumerate(files) if file.startswith(first_part)]

    return geom_fam_indices
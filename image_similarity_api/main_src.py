
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import torch 

import sys
import os

# Get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


from src.utils_predictor import dataset_feature_extr,img_feature_extr,cosine_similarity_np,geom_fam_mask_gen

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
            "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine.")
    import sys; sys.exit()
else:
    device = torch.device("mps")
    print("MPS device set")

################################################################################
################################################################################
model_type = "vgg" # Define the type of model being used
use_geometric_family = False # Flag indicating whether to use geometric family for image selection
file = "Circle_0a0b51ca-2a86-11ea-8123-8363a7ec19e6.png" # File name of the image
################################################################################
################################################################################
def main():

    # Importing necessary libraries
    import os
    import numpy as np
    from PIL import Image

    # Define parent directory and data directory paths
    parent_directory = os.path.abspath(os.pardir)
    data_directory = os.path.join(parent_directory, 'image_similarity_api/data')
    data_folder = os.path.join(data_directory,"2D geometric shapes dataset/output")
    
    batch_size = 64  # Batch size for processing
    
    file_path = "features.npy"
    # Check if features file exists, if not, generate features
    if not os.path.exists(file_path):
        dataset_feature_extr(data_folder, model_type, batch_size, device)
    
    # Load the image
    image_path = os.path.join(data_folder,file)
    img = Image.open(image_path)
    
    # Extract features from the image
    img_features = img_feature_extr(img,model_type,device)
    
    # Load precomputed features of all images
    images_features = np.load("features.npy")

    # If using geometric family, generate mask to filter images
    if use_geometric_family == True:
        geom_fam_indices = geom_fam_mask_gen(file,data_folder)
        images_features = images_features[geom_fam_indices]
    
    # Compute cosine similarity between the input image and all other images
    cos_sim = cosine_similarity_np(img_features, images_features)

    # Sort the cosine similarities and get indices of top 20 most similar elements
    top_indices = np.argsort(cos_sim.squeeze())[::-1][1:21]
    top_cos_similarities = cos_sim.squeeze()[top_indices]

    # Print the indices and cosine similarities of the top 20 most similar elements
    print("Top 20 Most Similar Elements:")
    for i, (index, similarity) in enumerate(zip(top_indices, top_cos_similarities), 1):
        print(f"{i}. Index: {index}, Cosine Similarity: {similarity}")
    

if __name__ == '__main__':
    main()

 
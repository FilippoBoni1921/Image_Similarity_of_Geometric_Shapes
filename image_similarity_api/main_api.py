
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import torch
from PIL import Image
from torchvision import transforms
import io
import os
import numpy as np

from src.utils_predictor import dataset_feature_extr,img_feature_extr,cosine_similarity_np

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
batch_size = 64 #batch size
################################################################################
################################################################################


# Load the dataset of 90,000 images
dataset_path = "data/2D geometric shapes dataset/output"
dataset_files = sorted(os.listdir(dataset_path))


file_path = "features.npy"
if not os.path.exists(file_path):

    parent_directory = os.path.abspath(os.path.join("src", os.pardir))
    data_directory = os.path.join(parent_directory, 'data')
    data_folder = os.path.join(data_directory,"2D geometric shapes dataset/output")

    dataset_feature_extr(data_folder, model_type, batch_size, device)

images_features = np.load("features.npy")

# Create a FastAPI instance
app = FastAPI()

# Endpoint to handle POST requests to "/predict"
@app.post("/predict")

async def predict(target_image: UploadFile = File(...)):
    try:
        # Read the content of the target image
        target_image_content = await target_image.read()
        
        # Open and preprocess the target image
        target_img = Image.open(io.BytesIO(target_image_content)).convert("RGB")
    
    # Compute feature representation for the target image using the model
        with torch.no_grad():
            target_features = img_feature_extr(target_img,model_type,device)

        cos_sim = cosine_similarity_np(target_features, images_features)

        # Find the indices of the top 20 most similar images
        top_indices = np.argsort(cos_sim.ravel())[::-1][1:21]

        cos_sim = cos_sim.flatten()
        cos_sim = cos_sim.astype(float)
        # Extract filenames and cosine similarity values based on indices
        similar_image_data = [{"filename": dataset_files[i], "cosine_similarity": float(cos_sim[i])} for i in top_indices]
        
        # Return only the filenames in the response
        return JSONResponse(content={"similar_image_filenames": similar_image_data})

    # Handle exceptions and return an error message
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

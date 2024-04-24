# Image Similarity Algorithm for Geometrical Shapes

## Presentation

In this work we are presenting a simple algorithm for the image similarity task. In particular we consider geometric shape images, and the goal is to give return the top 20 most similar images 
in the dataset to the input image. We also present a simple API built with FastAPI. The dataset is very large so it is not possible to find it in the repository. It can be found at the following
link: https://data.mendeley.com/datasets/wzr2yv7r53/1 .

### The Dataset
The Dataset is composed by more or less 90 000 images of different geometric shapes. The dataset does not have labels. The only info we have is the geometric shape represented in each image.

### The Model and its usage
As already said, the dataset doesn not provide any labels. So we do not have a ground truth indicating the most similar images for a given input image. Thus the task turns into an unsupervised task.\\
The main idea of the solution proposed is to avoid training a new deep learning algorith and used a pretrained model instead. This model would be used to extract representative features given the input image. Then given a processed representation of the images we can compare them and find the most similar ones. In particular we considered two models:**VGG** and **ResNet18**. This two model were chosen because they have been trained on Imagenet.\\
In order to compare the extraced features and find the most similar images we use the **cosine similarity** metric:

Cosine Similarity = (A • B) / (||A|| * ||B||)

### Pipeline
The pipeline followed by the algorithm is the following:
1. Load the data and apply the transformations necessary to use them as input to the models
2. Use the model to extract the features from every image of the dataset and save them in `features.npy`
3. Given an input image, load it and apply the necessary transformations
4. Use the model to extract the features of the input image
5. Compare the features of the input image to all the features of the image of the dataset in `features.npy` using the cosine similarity
6. Present the filenames and the cosine similarities of the 20 most similar images
7. If `features.npy` already exists, the pipeline strats from step 3.


## Set up

### Dataset
First of download the dataset from the link and put the downloaded folder in directory called `data`, to be placed in the directory `image_similarity_api` 

### Environment 
Create a virtual environment using the file `requirements.txt`

### Run the code 
To run the code it is enough to navigate to the directory `image_similarity_api` and run `main_src.py`.
The parameters to set inside the `main_src.py` file are:
- `model_type` (values:`vgg`,`res`;default:`vgg`) : set which one of the models to use
- `use_geometric_family`(values:`True`,`False`): set whether to use the metadata of the geometric shape to restrict the number of the possible similar images
- `file` :  specify the file name (or path) of the input

### Run the API
To run the API we need to first navigate to the directory `image_similarity_api` . Then start the API with the follwoing code:

`uvicorn main_api:app --reload` 

Then open another terminal and run the command:
`curl -X POST -F "target_image=@path/to/the/input/image.png" http://127.0.0.1:8000/predict` 

The output of the code will be printed in the terminal.
The parameters to set inside the `main_src.py` file are:
- `model_type` (values:`vgg`,`res`;default:`vgg`) : set which one of the models to use

The option `use_geometric_family` was not included in the API so in this case we get the results that do not use the metadata

**IMPORTANT:** the `model_type` used to generate the features and the one specified in the two `main_.py` files must be the same 




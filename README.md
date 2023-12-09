# Cervical-Fracture

# Project: Cervical Spinal Fracture Classification
This project falls within the domain of artificial intelligence, specifically focusing on computer vision. In the sub-domain of deep learning and image classification, various architectures such as Deep CNNs, Inceptionv3, Resnet50V2, and Xception are employed. The primary application of this project is in the field of medical image classification, particularly in diagnosing cervical spinal fractures from CT scans. The utilization of these advanced architectures demonstrates the project's commitment to leveraging cutting-edge techniques in artificial intelligence to assist physicians in accurately identifying fractures in cervical vertebrae through the analysis of medical imaging data.

## Problem Statement 
Cervical fractures often result from high-energy trauma such as automobile crashes and falls. In elderly people, a fall from a chair or to the ground can cause a cervical fracture. A cervical fracture involves the fracture of any of the seven cervical vertebrae in the neck. Given that these vertebrae support the head and connect it to the shoulders and body, an immediate response to an injury is crucial, as it can have severe consequences. Injuries to the vertebrae can lead to temporary or permanent paralysis from the neck down and, in some cases, even death. Therefore, physicians often rely on radiographic studies, such as MRI or CT scans, to assess the extent of injuries. This project aims to leverage AI to assist physicians in determining whether a CT scan image displays a "fracture" in a vertebra or is "normal."

## Dataset Details
This dataset, named "Spine Fracture Prediction from C.T. Dataset," encompasses images of cervical CT scans categorized into Fractured and Normal classes, organized within Train and Test folders. The dataset, accessible via the [Spine Fracture Prediction Dataset link](https://www.kaggle.com/datasets/vuppalaadithyasairam/spine-fracture-prediction-from-xrays/code), boasts a size of 311 MB. It comprises two classes: Fracture and Normal, making it a binary classification task. The dataset's training subset involves 3800 images, distributed with 3040 for training and 760 for validation. Additionally, there are 400 images in the testing subset. This dataset serves as a foundational component for training and evaluating models in the medical image classification domain, specifically targeting cervical spinal fractures.

## Classification Metrics of the Final Model
Accuracy Score: The model achieved perfect accuracy on the training set (1.0000), a high accuracy on the validation set (0.9122), and a good accuracy on the test set (0.8605).
Loss: The training set exhibited minimal loss (0.0024), while the validation set had slightly higher loss (0.1695), and the test set had a moderate loss (0.3144).
Precision: The model achieved perfect precision on the training set (1.0000), a good precision on the validation set (0.8482), and high precision on the test set (0.9048).
Recall: The training set displayed perfect recall (1.0000), the validation set achieved perfect recall (1.0000), and the test set had a good recall (0.7950). This set of metrics indicates a case of overfitting, suggesting potential areas for model improvement and optimization. It's essential to scrutinize the model architecture, parameters, and dataset preprocessing to address this overfitting and enhance the model's generalization capabilities. Furthermore, it's advised not to deploy this model in a production environment due to the observed overfitting.

## Tools / Libraries
The tools and environment used for this project are as follows:

Languages: Python
Tools/IDE: Anaconda
Libraries: Keras, TensorFlow
Virtual Environment: pipenv
Python served as the primary programming language for implementing the project. The Anaconda distribution, a comprehensive platform for data science and machine learning, was chosen as the integrated development environment (IDE). The key machine learning libraries employed in the project were Keras and TensorFlow, offering robust support for developing deep learning models. The virtual environment for managing dependencies and isolating the project's environment was established using pipenv, ensuring a controlled and reproducible development setup.

## Run the Model

To execute the scripts and notebooks in this project, start by cloning the repository using the provided git command. Once cloned, navigate to the project directory in a terminal or command prompt. Activate the virtual environment using 'pipenv shell' or install pipenv and activate it with 'pip install pipenv' and 'pipenv shell' commands. After activating the virtual environment, install project files and dependencies by running 'pipenv install'. Finally, follow the provided instructions to train the model using 'python train.py', start the prediction service with 'python predict_flask.py', and test the service using 'python predict_test.py'.
To deploy the model as a web service, two approaches are presented: Using Waitress and Using Docker. For Waitress, follow the outlined steps from 1 to 4, then run the prediction service with Waitress using the command in a terminal. Test the prediction service by executing python predict_test.py in another terminal. For Docker, clone the directory into your workspace, adjust import statements in predict_flask.py, build and run the Docker application with commands. Test the prediction service with python predict_test.py in another terminal. After successful testing, revert the import changes in predict_flask.py.

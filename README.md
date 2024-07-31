Medicinal Leaf Classification

Overview

This project is a web application that uses deep learning to classify medicinal leaves based on their images. The application allows users to upload an image of a leaf, and then uses a trained model to predict the type of leaf and provide information about its medicinal properties.

Features

Image classification: The application uses a deep learning model to classify the uploaded image into one of several categories of medicinal leaves.
Medicinal information: Once the leaf is classified, the application provides information about its medicinal properties, including its scientific name, medicinal property, medicinal details, common growth location, and disclaimer.
User-friendly interface: The application has a simple and intuitive interface that allows users to easily upload images and view the classification results.
Requirements

Python 3.11 or later
Streamlit 1.10 or later
TensorFlow 2.8 or later
Keras 2.8 or later
NumPy 1.22 or later
PIL 9.0 or later
Base64 1.0 or later
Installation

Clone the repository using git clone https://github.com/your-username/medicinal-leaf-classification.git
Install the required dependencies using pip install -r requirements.txt
Run the application using streamlit run app.py
Usage

Open the application in a web browser by navigating to http://localhost:8501
Click on the "Choose a file" button to upload an image of a medicinal leaf
Click on the "Predict" button to classify the image and view the results
View the classification results, including the predicted class, confidence, and medicinal information
Model

The model used in this application is a deep learning model trained on a dataset of images of medicinal leaves. The model is  hybrid model a convolutional neural network (CNN) with the layers of Recurrent neural Network that uses transfer learning to leverage the features learned from a pre-trained model.

Dataset

The dataset used to train the model consists of images of medicinal leaves, along with their corresponding labels and medicinal information. link:D:\Identification_plant\dataset

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

Acknowledgments

This project was developed by Kanupriya , Amarpreet Kaur Rekhi,Vani Mittal, and Satyam Sangwan as a final year group project for the Bachelor of Technology in Computer Science and Engineering at Meerut Institute Of Engineering & Technology. The authors would like to thank their mentors and contributors for their help and support.
Research Paper

This project is based on the research paper "Medicinal Leaf Classification using Deep Learning" published in the Tuijin Jishu/Journal of Propulsion Technology ISSN: 1001-4055Vol. 45No. 2(2024)_. https://propulsiontechjournal.com/index.php/journal/article/view/5988/3959
Future Work

Improve the accuracy of the model by collecting more data and fine-tuning the hyperparameters
Add more features to the application, such as the ability to search for specific medicinal leaves or view the classification results in a table
Deploy the application to a cloud platform or server for wider accessibility

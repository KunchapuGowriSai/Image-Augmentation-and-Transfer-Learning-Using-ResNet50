Image Augmentation and Classification Using ResNet50
This project involves training a deep learning model using the ResNet50 architecture for image classification. The dataset is augmented with various transformations, and the model is fine-tuned to classify the images into different classes.

Table of Contents
Requirements
Dataset Preparation
Augmentation Techniques
Model Architecture
Training
Evaluation
Prediction
References
Requirements
To run the project, you'll need the following dependencies:

TensorFlow 2.x
Keras
OpenCV
NumPy
Matplotlib
Google Colab (Optional)
Drive integration (Optional)
Install the dependencies using the following command:


pip install tensorflow opencv-python numpy matplotlib
Dataset Preparation
The dataset should be organized into folders, with each folder representing a class. The images in each folder will be read, resized, and augmented before training.


/dataset
    /class1
        image1.jpg
        image2.jpg
    /class2
        image1.jpg
        image2.jpg
The dataset can be stored on Google Drive for easy access through Google Colab.

Augmentation Techniques
This project includes custom image augmentation techniques to increase the size of the dataset:

Rotation: The image is rotated randomly by 90, 180, or 270 degrees.
Brightness Adjustment: Randomly alters the brightness of the image by manipulating its HSV values.
Images are augmented and saved to a directory for later use in training.

Model Architecture
The model used in this project is based on the ResNet50 architecture, a pre-trained model on ImageNet. The model’s layers are frozen, and custom layers are added on top for image classification.

Model Layers:
Pre-trained ResNet50 (without top layers)
GlobalAveragePooling2D
Dense (512 units, ReLU)
Dense (128 units, ReLU)
Dense (Output layer for classification, softmax)
The ResNet50 model's weights are frozen to prevent retraining on the ImageNet dataset, and only the newly added layers are trained.

Training
The model is compiled with the Adam optimizer and sparse categorical crossentropy as the loss function. The model is trained using the augmented images, with a callback to stop training when the accuracy reaches 99%.


transferresnet.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
transferresnet.fit(images, labels, epochs=100, callbacks=[callbacks])
Evaluation
The model is evaluated using the same set of augmented images:


transferresnet.evaluate(images, labels)
Prediction
A custom prediction function is provided to predict the class of new images. The function takes an image, resizes it, and runs it through the trained model to output the predicted class.


def predict(image_path, model, labels):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    pred_class = np.argmax(model.predict(np.array([img])))
    print(f"Predicted class: {labels[pred_class]}")
Example usage:
predict("feather.jpg", transferresnet, classes)
References
ResNet50: https://keras.io/api/applications/resnet/
OpenCV: https://opencv.org/
TensorFlow Documentation: https://www.tensorflow.org/

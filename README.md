# Emotion_detection_with_CNN

### Packages need to be installed (tensorflow is available only till python 3.11)
- pip install numpy
- pip install opencv-python
- pip install keras
- pip install --upgrade tensorflow
- pip install pillow

### Dataset used:
This code uses below dataset from KAggle for training the models
https://www.kaggle.com/datasets/msambare/fer2013

### Train Emotion detector
- We have stored the model obtained after training the datasets under "data" folder inside "code" folder as due to space restrictions we could not add the whole dataset.
- In case the dataset has to be trained, kindly add the two folers "train" and "test" from the kaggle data to folder named "data" and run below commands
- commands:
- python CNNTraining.py
- python AugmentedDataCNN_Training.py

Running the training files generate below files under model folder
emotion_model.json
emotion_model.h5

### run emotion detection
- python FaceEmotionDetection.py (This file performs the emotion detection qith squares around the faces once detected and the name of the respective emotion)
- python PerformanceAnalysis.py (This file generates the confusion matrix and gives precision, F1-Score, Recall)


### Expected Output

Emotion Detection output will produce a window with the vide passed and a green square around the faces detected and the name of the emotion written on top of the square.
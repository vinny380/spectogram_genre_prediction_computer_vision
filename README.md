# Music Genre Classification with Deep Learning

## Overview:
- This project focuses on classifying music genres using Deep Learning techniques, particularly Convolutional Neural Networks (CNNs). Spectrograms or sonograph images of music were utilized to classify songs into seven distinct categories: Classical, Electronic, Folk, Hip-Hop, Pop, Punk, or Rock.

## How to Run Predictions:
- We've attached a small portion of the dataset (since it's so heavy) so you can experiment with the model. Each class has 10 samples.
- run ```python3 predict.py ResNetCustom.pth /relative/path/to/image```
Example:
Let's say you want to predict an Electronic song, with file name 000152.jpg, you'd run:
```python predict.py ResNetCustom.pth subsampled_data/Electronic/000152.jpg```
```python predict.py ResNetCustom.pth subsampled_data/Rock/004586.jpg```


## How To Run Training:
- Due to the nature of our project (high-definition images), we couldn't attach the training dataset to this.
- We tried turning it into numpy arrays and PyTorch tensors, but it still was 4.5 GB+.
- You can download the dataset here: https://drive.google.com/drive/folders/1jwuBe1-Fd7elNHS9RopqwFi3w2IFc1D1?usp=sharing
- Keep in mind, that this is 5 GB worth of images. After you download it, put all genre folders
into one folder called data and you're good to go.
- Feel free to use the create_subsampled_dataset() function to create a smaller sample of the dataset once you have it, so you don't go over ALL the images in the dataset.
- After that, simply re-run the Jupyter Notebook or run ```python3 main.py```
- Make sure you're in the right directory (Deep Learning)

## Dataset:
- The dataset used for this task was the POG Music Spectrogram dataset, which contains spectrogram images for 19 different music genres. However, due to poor dataset management, only the training set was utilized, consisting of approximately 17,500 samples. The dataset lacked labels for the training set, posing a challenge for supervised learning tasks. Hence we removed extremely underepresented classes, going from 19 to 7 classes, due to computational resources available (i.e. lack of GPU).

## Setup & Reproducibility:
- The project was conducted on an M1 MacBook Air initially, but computational limitations led to the utilization of Google Colab Pro with an NVIDIA A100 Tensor GPU. To ensure reproducibility, a randomized seed was set using np.random.seed(123).

## Data Preprocessing:
- The chosen CNN architecture was a customized version of the ResNet50 model, adapted for classifying images into seven categories. Data preprocessing involved several transformations using the torchvision.transforms module, including random resized cropping, normalization, and conversion to tensors.

## Model:
- The ResNet50 model was customized with a dropout layer (50% dropout rate) before the final fully connected layer to mitigate overfitting. A stochastic Gradient Descent (SGD) optimizer with weight decay was used for training and a learning rate scheduler optim.lr_scheduler.ReduceLROnPlateau was employed to adjust the learning rate based on validation performance.

## Technique:
- Forward propagation involved breaking the input image into pixel matrices and passing them through convolutional layers for feature extraction. Loss calculation utilized cross-entropy, and backpropagation updated model weights via SGD. Learning rate scheduling and regularization techniques were employed to enhance training efficiency and prevent overfitting.

## Weaknesses & Contingency Plan:
- Computational limitations and dataset complexities posed challenges, compromising dataset size and model performance. Future efforts will focus on refining hyperparameters, exploring alternative data augmentation techniques, and experimenting with different architectures to improve classification accuracy.

## Prerequisites:
Summed up in the requirements.txt

## Installation:
pip3 install -r requirements.txt

## Results:
The performance of the CNN model is evaluated using metrics such as accuracy and cross-entropy loss on both the training and validation datasets. Our test set got 54%, with out best model saved by the best k-th fold having 61% accuracy. The latter is also the model being used for prediction in predict.py

## Acknowledgements
Dataset: https://www.kaggle.com/datasets/gauravduttakiit/python-pog-spectogram-music-classification

# Simple IMDB BERT

This repository contains the code for training a BERT model from scratch. 

**Note:**
This code is to be used for demonstration purposes and a pre-trained BERT 
model would be far more effective for downstream tasks. 

An interactive Jupyter notebook has been developed and the code has also been refactored 
into Python scripts.

## Setup

At the time of development, tensorflow-text was unable to be fully installed 
onto my local machine and was developed on a Google Colab instance.

To setup the instance:

```shell
!apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
!pip uninstall -y -q tensorflow keras tensorflow-estimator tensorflow-text
!pip install -q -U tensorflow-text tensorflow
```
This code was tested with ```tensorflow==2.11.0``` and ```tensorflow-text==2.11.0```.

## Python Scripts

To train the model using the Python scripts, the following steps need to be run in order:

1. download.sh
2. data_prep.py
3. tokeniser_build.py
4. model_training.py

```shell
./download.sh
python3 data_prep.py
python3 tokeniser_build.py
python3 model_training.py
```

On a GPU enabled colab instance the model took approximately 10mins per epoch to train.
On the completion of each callback, the model outputs predicted tokens for the example phrase:

*I watched this **[MASK]** and it was awesome*

Example output during training process:

```commandline
Epoch 10/10
1/1 [==============================] - 0s 23ms/step
{'input_text': 'i have watched this [MASK] and it was awesome', 'prediction': 'i have watched this movie and it was awesome', 'probability': 7.254052, 'predicted mask token': 'movie'}
{'input_text': 'i have watched this [MASK] and it was awesome', 'prediction': 'i have watched this saw and it was awesome', 'probability': 6.222549, 'predicted mask token': 'saw'}
{'input_text': 'i have watched this [MASK] and it was awesome', 'prediction': 'i have watched this i and it was awesome', 'probability': 5.614587, 'predicted mask token': 'i'}
{'input_text': 'i have watched this [MASK] and it was awesome', 'prediction': 'i have watched this really and it was awesome', 'probability': 5.174042, 'predicted mask token': 'really'}
{'input_text': 'i have watched this [MASK] and it was awesome', 'prediction': 'i have watched this a and it was awesome', 'probability': 5.1586485, 'predicted mask token': 'a'}
524/524 [==============================] - 453s 865ms/step - loss: 0.1523 - masked_accuracy: 0.9771 - val_loss: 0.2180 - val_masked_accuracy: 0.9754
```
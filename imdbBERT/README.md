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


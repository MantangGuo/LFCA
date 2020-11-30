# LFCA
PyTorch implementation of ECCV 2020 paper (Oral): "[Deep Spatial-angular Regularization for Compressive Light Field Reconstruction over Coded Apertures](https://arxiv.org/abs/2007.11882)".

## Requrements
- Python 3.7.4
- PyTorch 1.3.1
- Matlab (for training/test data generation)

## Dataset
We provide MATLAB code for preparing the training and test data. Please first download light field datasets, and put them into corresponding folders in LFData.

## Test
We provide the pre-trained model for 2 -> 49 task on Lytro dataset. 

To test, run:
```
python lfca_test.py
```

## Train
To train, run:
```
python lfca_train.py
```

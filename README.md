# Recomender System for MovieLens 1M Dataset

## Prerequisites
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN 

## Getting started

- Clone this repo:
```bash
git clone https://github.com/yilmazgencc/Recommender_System.git
cd Recommender_System
```

- Install PyTorch 3.7 and other dependencies via

- For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.

- For pip users, please type the command `pip install -r requirements.txt`.

- Train the UC model with default parameters:

```bash
python train.py
```
- Train the UC model with custom parameters:
```bash
python train.py --model $MODEL  --opt $OPTIMIZER  --embedding_dim $EMBEDDING_DIMENSION --epochs $EPOCH_NUMBER 
--batch_size $BATCH_SIZE --lr $LEARNING_RATE --dropout $DROPOUT_RATE
```
- The list arguments is as follows:
- --model: Designed neural network(NN) and confusion matrix(CF) available default="NN"
- --opt: Available optimizer for training( "Adam", "SGD", "RMSprop", "Adagrad") default="Adam"
- --embedding_dim:Embedding layer dimension default=64
- --epochs: Epoch number for training default=25
- --batch_size: Batch size selection default=1024
- --lr:Learning rate for training  default=1e-4
- --dropout: Droput rate for NN default=0.4
- --seed: Random seed number  default=123
- --use_deterministic: Seed activation option default="store_true"
- --specific_user: Specific user pretiction option activation  default="store_true"
- --user_id: Specific user ID for prediction
- --train_test_split: Train and test split ratio for training default=0.8

### Results and Figures

Results:

NN with 3 layers lr:1e-4
- best validation loss: 0.7788
- best validation rmse: 0.8825

<p align="center">
 <img src="imgs/NN_loss.png" width="400px"/> | <img src="imgs/NN_rmse.png" width="400px"/>
</p>

CF with 3 layers
- best validation loss: 0.7403
- best validation rmse: 0.8604


<p align="center">
 <img src="imgs/CF_loss.png" width="400px"/> | <img src="imgs/CF_rmse.png" width="400px"/>
</p>

## Issues

- Please report all issues on the public forum.

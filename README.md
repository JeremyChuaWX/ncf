# neural-collaborative-filtering

Neural collaborative filtering(NCF), is a deep learning based framework for making recommendations. The key idea is to learn the user-item interaction using neural networks. Check the follwing paper for details about NCF.

> He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2017.

## Dataset

[Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

## Files

> `preprocess.py`: preprocesss Kaggle data
>
> `data.py`: prepare train/test dataset
>
> `utils.py`: some handy functions for model training etc.
>
> `metrics.py`: evaluation metrics including hit ratio(HR) and NDCG
>
> `gmf.py`: generalized matrix factorization model
>
> `mlp.py`: multi-layer perceptron model
>
> `cnn.py`: convolutional neural network model
>
> `neumf.py`: fusion of gmf, mlp and cnn
>
> `engine.py`: training engine
>
> `train.py`: entry point for train a NCF model

## Instructions

- Download the raw data from [Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

- Create the folders

  - `./src/data/raw`
  - `./src/data/processed`
  - `./src/checkpoints`

- Extract the raw data to `./src/data/raw`

- Run `preprocess.py` to preprocess the data

- If CUDA is available or using Apple Scilicon, u can enable the gpu flags in the configs (found in `./src/config.py`) accordingly to speed up training

  - `use_cuda`
  - `use_mps`

- Run `train.py` with flags to pretrain the individual models

  ```bash
  python3 train.py --model="gmf" --data="<processed data file>"
  python3 train.py --model="mlp" --data="<processed data file>"
  python3 train.py --model="cnn" --data="<processed data file>"
  ```

- Replace the filenames (from `./src/checkpoints`) in `neumf_config` to load the pretrained model weights.
  `neumf_config` can be found in `./src/configs.py`, edit the entries

  - `pretrain_mf`
  - `pretrain_mlp`
  - `pretrain_cnn`

- Run `train.py` with the "neumf" flag to train the final ensemble NCF

  ```bash
  python3 train.py --model="neumf" --data="<processed data file>"
  ```

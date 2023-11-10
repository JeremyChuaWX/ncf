# neural-collaborative-filtering

Neural collaborative filtering(NCF), is a deep learning based framework for making recommendations. The key idea is to learn the user-item interaction using neural networks. Check the follwing paper for details about NCF.

> He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2017.

## Dataset

[The Movielens 1M Dataset](http://grouplens.org/datasets/movielens/1m/) is used to test the repo.

## Files

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
> `neumf.py`: fusion of gmf and mlp
>
> `engine.py`: training engine
>
> `train.py`: entry point for train a NCF model

## Instructions

- Download the raw data from [Kaggle](<>)

- Extract the raw data to `./src/data/raw`

- Create a folder `./src/data/processed`

- Run `preprocess.py` to preprocess the data

- Run `train.py` with flags to pretrain the individual models

  ```bash
  python3 train.py -- --model="<model name here>"
  ```

- Replace the filenames in `neumf_config` to point to the latest model states

- Run `train.py` with the "neumf" flag to train the final ensemble NCF

  ```bash
  python3 train.py -- --model="neumf"
  ```

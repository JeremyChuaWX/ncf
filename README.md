# neural-collaborative-filtering

## Dataset

[MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)

## Files

- `preprocess.py`: preprocesss Kaggle data
- `data.py`: prepare train/test dataset
- `train.py`: entry point for train a NCF model
- `utils.py`: some handy functions for model training etc.
- `metrics.py`: evaluation metrics including hit ratio(HR) and NDCG
- `gmf.py`: generalized matrix factorization model
- `mlp.py`: multi-layer perceptron model
- `cnn.py`: convolutional neural network model
- `neumf.py`: fusion of gmf, mlp and cnn
- `engine.py`: training engine

## Data Preprocessing

- Download the raw data from [grouplens](https://grouplens.org/datasets/movielens/1m/)

- Create the folders

  - `./src/data/raw`
  - `./src/data/processed`
  - `./src/checkpoints`

- Extract the raw data to `./src/data/raw`

  - The raw data folder should contain the file `ratings.dat` as such: `./src/data/raw/ratings.dat`

- Run `preprocess.py` to preprocess the data

  - The script will preprocess the raw `ratings.dat` file and store it in a csv file `./src/data/processed/ratings.csv`

## Model Configuration

- All model configurations are kept in `./src/config.py` in the `get_configs` function which is used to initialise all model configs

- If CUDA is available or using Apple Scilicon, you can enable the gpu flags in the configs to speed up training

  ```python
  def get_configs(num_user, num_item):
    base_config = {
      "use_cuda": False, # set to true if CUDA is available
      "use_mps": False, # set to true if using Apple Scilicon and Metal API is available
    }
    # ...
  ```

## Model Training

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

## References

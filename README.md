# Rationalizing Fact-Checks

## General instructions.

Install required dependencies:
```
pip install -r requirements.txt
```

Download and process data following `README.md` in `[DATA_NAME]` folder:
```
cd data/[DATA_NAME]
```

Train models or analyze rationales with `run.py`:
```
python rationalize/run.py --mode=[MODE] --data_name=[DATA_NAME] --config_name=[CONFIG_NAME]
```

`[MODE]`:
- `train`: train a model.
- `evaluate`: evaluate a model.
- `output`: output rationales.
- `binarize`: binarize rationales to 0/1 (soft rationalization only).
- `vectorize`: generate vectors/embeddings for rationales.
- `cluster`: cluster rationales and plot figures.

`[DATA_NAME]`:
- `movie_reviews`: the dataset of movie reviews.
- `personal_attacks`: the dataset of fact-checks.
- `fact-checks`: the dataset of fact-checks.
- `glove`: pretrained GloVe embeddings.

`[CONFIG_NAME]`:
- e.g., `soft_rationalizer` or any `.config` files in `[DATA_NAME]` folder.

## Instructions for replicating results in the paper.

### Replicating results for Table 1.

Here is the instruction to replicate the `movie_reviews` column of Table 1. To replicate another column simply replace `movie_reviews` to `personal_attacks` in all the command lines.

First make sure that the dataset and embeddings are prepared: 
```
cd data/movie_reviews
./prepare_data.sh
cd ../..
cd data/glove
./prepare_data.sh
cd ../..
```

Then, run the following command, each line corresponds to an experiment from h0-h3 and s0-s1:
```
python rationalize/run.py --mode=train --data_name=movie_reviews --config_name=soft_rationalizer
```

To get the results for s2-s3, switch to the data folder and run:
```
python
```

### Replicating results for Figures 3-5.

We have logged data to plot figures 3-5.

To plot figure 3, run:
```
python rationalize/run.py --mode=cluster --data_name=fact-checks --config_name=soft_rationalizer_w_domain
```

To plot figures 4-5, run:
```
cd data/fact-checks
python 
```

If you would like to train the model from scratch, run the following command in sequence.
```
cd 
```





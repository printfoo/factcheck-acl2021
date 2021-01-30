# Rationalizing Fact-Checks

## General instructions.

1. Install required dependencies:
```
pip install -r requirements.txt
```

2. Download and process data following `README.md` in `[DATA_NAME]` folder:
```
cd data/[DATA_NAME]
```

3. Train models or analyze rationales with `run.py`:
```
python rationalize/run.py --mode=[MODE] --data_name=[DATA_NAME] --config_name=[CONFIG_NAME]
```

`[MODE]`:
- `train`: train a model.
- `eval`: evaluate a model.
- `output`: output rationales.
- `binarize`: binarize rationales to 0/1 (soft rationalization only).
- `vectorize`: generate vectors/embeddings for rationales.
- `cluster`: cluster rationales and plot figures.

`[DATA_NAME]`:
- `movie_reviews`: a dataset of movie reviews.
- `personal_attacks`: a dataset of fact-checks.
- `fact-checks`: a dataset of fact-checks.

`[CONFIG_NAME]`:
- e.g., `soft_rationalizer` or any `.config` files in `[DATA_NAME]` folder.

## Instructions for replicating results in the paper.





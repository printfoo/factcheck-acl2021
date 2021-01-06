# Rationalization.

## Instructions.

Download data:
```
$ cd data/[DATASET]/raw
$ ./download.sh
```

Preprocess data:
```
$ cd data/[DATASET]
$ python data_cleaner.py
$ python data_signaler.py
```

Train a model:
```
$ python rationalize/run.py 
```

# NLP rationalization.

## Instructions.

Download and clean data:
```
$ cd data/[DATASET]/raw
$ ./download.sh
$ cd ..
$ python data_cleaner.py
```

Train a model:
```
$ python rationalize/run.py [PARAMS]
```

## References.

Part of this repo is built on:  
- Mo Yu, Shiyu Chang, Yang Zhang, Tommi S. Jaakkola, **Rethinking Cooperative Rationalization: Introspective Extraction and Complement Control**, *EMNLP 2019* (https://github.com/Gorov/three_player_for_emnlp).
- The script was updated to Python3 by Abishkar Chhetri (https://github.com/chhetri22/three_player_for_emnlp).

# Personal Attacks Dataset.

Run the following script to prepare the dataset:
```
./prepare_data.sh
```

This script contains the following steps:

1. Download data and lexicon:
```
cd raw
wget "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/7554634/attack_annotated_comments.tsv"
wget "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/7554637/attack_annotations.tsv"
wget "https://raw.githubusercontent.com/shcarton/rcnn/master/deliberativeness/data/processed/wiki/personal_attacks/wiki_attack_dev_rationale.csv"
wget "https://raw.githubusercontent.com/shcarton/rcnn/master/deliberativeness/data/processed/wiki/personal_attacks/wiki_attack_test_rationale.csv"
wget "https://raw.githubusercontent.com/uds-lsv/lexicon-of-abusive-words/master/Lexicons/baseLexicon.txt"
cd ..
```

2. Clean data:
```
python data_cleaner.py
```

3. Add domain knowledge:
```
python data_signaler.py
```

The dataset was originally released by:  
- Ellery Wulczyn, Nithum Thain, Lucas Dixon, **Ex Machina: Personal Attacks Seen at Scale**, *WWW 2017* (https://figshare.com/articles/Wikipedia_Talk_Labels_Personal_Attacks/4054689).

Rationale annotations and part of data cleaning script was released by:
- Samuel Carton, Qiaozhu Mei, Paul Resnick, **Extractive Adversarial Networks: High-Recall Explanations for Identifying Personal Attacks in Social Media Posts**, *EMNLP 2018* (https://github.com/shcarton/rcnn).

The lexicon was released by:
- Michael Wiegand, Josef Ruppenhofer, Anna Schmidt, Clayton Greenberg, **Inducing a Lexicon of Abusive Words - A Feature-Based Approach**, *NAACL 2018* (https://github.com/uds-lsv/lexicon-of-abusive-words).

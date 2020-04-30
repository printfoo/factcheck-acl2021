# Personal Attacks Dataset.

Download raw dataset to the `/raw` folder:
```
https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/7554634/attack_annotated_comments.tsv
https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/7554637/attack_annotations.tsv
https://raw.githubusercontent.com/shcarton/rcnn/master/deliberativeness/data/processed/wiki/personal_attacks/wiki_attack_dev_rationale.csv
https://raw.githubusercontent.com/shcarton/rcnn/master/deliberativeness/data/processed/wiki/personal_attacks/wiki_attack_test_rationale.csv
```

Clean data at this directory `.`:
```
$ python data_cleaner.py
```

Download abusive word lexicon to the `/raw` folder:
```
https://raw.githubusercontent.com/uds-lsv/lexicon-of-abusive-words/master/Lexicons/baseLexicon.txt
```

Add linear siganl and domain knowledge at this directory `.`:
```
$ python data_signaler.py
```

The dataset was originally released by:  
- Ellery Wulczyn, Nithum Thain, Lucas Dixon, **Ex Machina: Personal Attacks Seen at Scale**, *WWW 2017* (https://figshare.com/articles/Wikipedia_Talk_Labels_Personal_Attacks/4054689).

Additional rationale annotations and part of data cleaning script was developed by:
- Samuel Carton, Qiaozhu Mei, Paul Resnick, **Extractive Adversarial Networks: High-Recall Explanations for Identifying Personal Attacks in Social Media Posts**, *EMNLP 2018* (https://github.com/shcarton/rcnn).

The lexicon was released by:
- Michael Wiegand, Josef Ruppenhofer, Anna Schmidt, Clayton Greenberg, **Inducing a Lexicon of Abusive Words - A Feature-Based Approach**, *NAACL 2018* (https://github.com/uds-lsv/lexicon-of-abusive-words).

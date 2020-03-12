# Personal Attacks Dataset.

Download raw dataset at:
```
https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/7554634/attack_annotated_comments.tsv
https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/7554637/attack_annotations.tsv
https://github.com/shcarton/rcnn/raw/master/deliberativeness/data/processed/wiki/personal_attacks/wiki_attack_dev_rationale.csv
https://github.com/shcarton/rcnn/blob/master/deliberativeness/data/processed/wiki/personal_attacks/wiki_attack_test_rationale.csv
```

Clean data *at this directory*:
```
python data_cleaner.py
```

The dataset was originally released by:  
```
Ellery Wulczyn, Nithum Thain, Lucas Dixon, "Ex Machina: Personal Attacks Seen at Scale", WWW 2017
```

Additional rationale annotations and part of data cleaning script was developed by:
```
Samuel Carton, Qiaozhu Mei, Paul Resnick, "Extractive Adversarial Networks: High-Recall Explanations for Identifying Personal Attacks in Social Media Posts", EMNLP 2018
```
# Rationalizing Fact-Checks

## General instructions.

1. Install required dependencies:
```
pip install -r requirements.txt
```

2. Download and process data following instructions in data folders:
```
cd data/[DATA_NAME]
```

3. Train/evaluate/etc models or output/cluster/etc rationales:
```
python rationalize/run.py --mode=[MODE] --data_name=[DATA_NAME] --config_name=[CONFIG_NAME]
```

## Instructions for replicating results in the paper.


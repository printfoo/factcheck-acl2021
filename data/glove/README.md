# GloVe Embeddings.

Run the following script to prepare the embeddings:
```
./prepare_data.sh
```

This script contains the following steps:

1. Download pre-trained embeddings and unzip it:
```
cd raw
wget "http://nlp.stanford.edu/data/glove.6B.zip"
unzip glove.6B.zip
mv glove.6B.*d.txt ../
cd ..
```

The dataset was released by:
- Jeffrey Pennington, Richard Socher, and Christopher D. Manning, **GloVe: Global Vectors for Word Representation**, *EMNLP 2014* (https://nlp.stanford.edu/projects/glove).

# Movie Reviews Dataset.

Run the following script to prepare the dataset:
```
./prepare_data.sh
```

This script contains the following steps:

1. Download data and lexicon:
```
cd raw
wget "http://www.eraserbenchmark.com/zipped/movies.tar.gz"
tar -xvf "movies.tar.gz"
wget "https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip"
unzip "NRC-Emotion-Lexicon.zip"
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
- Bo Pang, Lillian Lee, Shivakumar Vaithyanathan, **Thumbs up? Sentiment Classification using Machine Learning Techniques**, *EMNLP 2002* (http://www.cs.cornell.edu/people/pabo/movie-review-data).

Rationale annotations was released by:
- Omar F. Zaidan, Jason Eisner, Christine Piatko, **Using "Annotator Rationales" to Improve Machine Learning for Text Categorization**, *NAACL 2007* (http://www.cs.jhu.edu/~ozaidan/rationales).

Additional annotations was released by:
- Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, Eric Lehman, Caiming Xiong, Richard Socher, Byron C. Wallace, **ERASER: A Benchmark to Evaluate Rationalized NLP Models**, *arXiv 2020* (http://www.eraserbenchmark.com).

The lexicon was released by:
- Saif Mohammad, Peter Turney, **Crowdsourcing a Word-Emotion Association Lexicon**, *Computational Intelligence 2013* (https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm).

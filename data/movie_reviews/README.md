# Movie Reviews Dataset.

Download raw dataset to the `/raw` folder:
```
http://www.eraserbenchmark.com/zipped/movies.tar.gz
```

Clean data at this directory `.`:
```
$ python data_cleaner.py
```

Download emotional lexicon EmoLex to the `/raw` folder:
```
http://sentiment.nrc.ca/lexicons-for-research/NRC-Emotion-Lexicon.zip
```

Add linear siganl and domain knowledge at this directory `.`:
```
$ python data_signaler.py
```

Evaluate linear siganl and domain knowledge based rationales:
```
$ python data_evaluator.py
```

The dataset was originally released by:  
- Bo Pang, Lillian Lee, Shivakumar Vaithyanathan, **Thumbs up? Sentiment Classification using Machine Learning Techniques**, *EMNLP 2002* (http://www.cs.cornell.edu/people/pabo/movie-review-data).

Rationale annotations was released by:
- Omar F. Zaidan, Jason Eisner, Christine Piatko, **Using "Annotator Rationales" to Improve Machine Learning for Text Categorization**, *NAACL 2007* (http://www.cs.jhu.edu/~ozaidan/rationales).

Additional annotations was released by:
- Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, Eric Lehman, Caiming Xiong, Richard Socher, Byron C. Wallace, **ERASER: A Benchmark to Evaluate Rationalized NLP Models**, *arXiv 2020* (http://www.eraserbenchmark.com).

The lexicon was released by:
- Saif Mohammad, Peter Turney, **Crowdsourcing a Word-Emotion Association Lexicon**, *Computational Intelligence 2013* (https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm).

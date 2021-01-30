# Fact-Checks Dataset.

To protect the copyright of Snopes.com, we provide only URLs to fact-checks in `raw`.

To get this dataset, first download webpages one by one from Snopes.com:
```
python data_downloader.py
```
This could take several hours depending on your internet speed.

Then, extract text from HTML webpages:
```
python data_extractor.py
```

Clean data:
```
python data_cleaner.py
```

Train word2vec:
```
python data_word2vec.py
```

After outputing rationales, filter them.
```
python rationale_filterer.py
```

Map rationales to fact-checks:
```
python rationale_mapper.py
```

Plot results:
```
python result_visualizer.py
```

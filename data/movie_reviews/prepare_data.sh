cd raw
wget "http://www.eraserbenchmark.com/zipped/movies.tar.gz"
tar -xvf "movies.tar.gz"
wget "https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip"
unzip "NRC-Emotion-Lexicon.zip"
cd ..
python data_cleaner.py
python data_signaler.py

## Setting up data

Follow directions in data_helpers/download_reddit_comment_commands.txt to download Reddit data. Then go through the following:

Assume you are in src directory

Make directory for data: ```mkdir data```

Make directory for websites (needed for Pyserini): ```mkdir data/websites```

Make directory for output (BM25): ```mkdir out/bm25_runs```

Make directory for output (lstm): ```mkdir out/lstm_runs```

Make directory for output (semantic): ```mkdir out/semantic_runs```

Navigate to data helpers directory: ```cd data_helpers```

Set up comment structure: ```python3 format_reddit_comments.py --raw ../data/RC_2009-05 --out ../data/RC_2009-05_formatted.json```

Scrape URLs: ```python3 scrape_urls.py --formatted ../data/RC_2009-05_formatted.json --out ../data/RC_2009-05_webpages.json```

## Running baselines

Assume you are in src directory

Run BM25 baseline: use bm25_baseline.ipynb

Run Semantic baseline: ```python3 neural.py --model semantic_baseline --relevance_scores data/relevance_scores.txt --corpus data/encoded_websites.pkl --src_train data/encoded_queries_train.pkl --src_dev data/encoded_queries_dev.pkl --src_test data/encoded_queries_test.pkl --out out/semantic_runs/```

Run LSTM: ```python3 neural.py --model lstm --relevance_scores data/relevance_scores.txt --corpus data/encoded_websites.pkl --src_train data/encoded_queries_train.pkl --src_dev data/encoded_queries_dev.pkl --src_test data/encoded_queries_test.pkl --out out/lstm_runs/```

Evaluate: ```python3 -m pyserini.eval.trec_eval -m map -m P.1 data/relevance_scores.txt <path to run>```



## TODO
- [ ] Update lstm.py and neural.py with proper data loading, batch size
- [ ] Add more domains to the data set
- [ ] Try with newer reddit comments
- [ ] Add data parallel wrappers
- [ ] Use model files and place all hyper parameters in there
- [ ] Semantic search baseline
- [ ] Transformer performance on dev
- [ ] Combine LSTM test method with semantic search (lots of reused code)
- [ ] Make test functions search over faiss instead of loading everything at once (only a problem with scale?)
- [ ] Use a larger, more powerful encoder (and train it?)
- [ ] Negative samples with seed for LSTM may not be random, instead do it over batch


# Context and Connections
An analysis of hyperlink mentions on the Internet.

# THIS FILE IS OUT OF DATE - NEED TO UPDATE

## Data file structure
You'll need to make the directories (starred)
```
data_2017-09
|____**encoded_queries**
|   |   queries_{train,val,test}.pkl
|____**encoded_webpages**
|   |   webpages.pkl
|____**pyserini**
|   |   files automatically created when bm25_baseline.ipynb is ran
|____**queries**
|   |   queries_{train,val,test}.tsv
|____**webpages**
|   |   {webpage_id}.json
|____all_needed_comments.pkl
|____all_urls.pkl
|____RC_2017-09
|____url_file_map.pkl
|____valid_urls.pkl
```
## Setting up data

Navigate to data helpers directory: ```cd data_helpers```

Follow directions in setup.txt to download Reddit data. Then go through the following:

Set up comment structure: ```python3 format_reddit_comments.py --out_path ../data_2017-09/ --raw_reddit_data ../data_2017-09/RC_2017-09```

This will create 
1. ```all_needed_comments.pkl``` A dictionary that maps comment id to raw comment body. Created based on the URLs found and their respective paths back to the root comment.
2. ```all_urls.pkl``` A dictionary that maps line number (corresponding to a comment, from the original reddit comment json file) to ```{'urls': [all urls in the comment], 'id': id of comment, 'chain': chain of ids from root to comment}```
3. ```valid_urls.pkl``` Same as ```all_urls.pkl```, but only for URLs that pass the selection criteria. 

Now, we can use the ids in the chains of ```valid_urls.pkl``` and ```all_needed_comments.pkl``` to reconstruct comment threads

Scrape URLs: ```python3 scrape_urls.py --min_length 2 --data_path ../data_2017-09/```

This will create
1. A collection of scraped webpage text in the ```webpages``` directory (created in setup.txt). Each webpage will have a unique file name and be formatted as a json (matching the expected input for pyserini)
2. ```url_file_map.pkl``` A dictionary that maps a URL to ```{'filename': name of file in webpages, 'chains': list of comment chains ending with that url, 'status': whether or not the page was successfully scraped, 'split': train, val, or test}```

Next, we can build our queries and relevance score file: ```python3 build_queries.py --data_path ../data_2017-09/ --out ../data_2017-09/queries/```

This will create
1. A train/val/test tsv split of plaintext comment chains in the ```queries``` directory (created in setup.txt). Each comment will be separated by a <C> tag.
2. ```relevance_scores.txt``` in the ```queries``` directory. Each line maps a query id to a relevant webpage filename. 


To pre-encode the queries and webpages for the semantic baseline and the preencoded lstm:

```python3 encode_dataset.py --query_path ../data_2017-09/queries/queries_train.tsv --out ../data_2017-09/encoded_queries/queries_train.pkl```

```python3 encode_dataset.py --query_path ../data_2017-09/queries/queries_val.tsv --out ../data_2017-09/encoded_queries/queries_val.pkl```

```python3 encode_dataset.py --webpages_path ../data_2017-09/webpages/ --out ../data_2017-09/encoded_webpages/webpages.pkl```


## Training and evaluating the models

Assume you are in src directory

Run BM25 baseline: use bm25_baseline.ipynb

Run Semantic baseline: ```python3 semantic_baseline.py --corpus data_2017-09/encoded_webpages/webpages.pkl --queries data_2017-09/encoded_queries/queries_train.pkl --out out/semantic_runs/```

Run LSTM preencoded: ```python3 lstm_preencoded.py --relevance_scores data_2017-09/queries/relevance_scores.txt --corpus data_2017-09/encoded_webpages/webpages.pkl --queries_train data_2017-09/encoded_queries/queries_train.pkl --queries_val data_2017-09/encoded_queries/queries_val.pkl --out out/lstm_preencoded_runs/```

Run Semantic finetune: ```python3 semantic_finetune.py --corpus data_2017-09/webpages/ --queries data_2017-09/queries/```

Evaluate: ```python3 -m pyserini.eval.trec_eval -m map -m P.1 <path to relevance scores> <path to run>```



## TODO
- [x] Add more domains to the data set
- [x] Try with newer reddit comments
- [x] Address scrape error to restrict scraped domains to only those selected, not ones present in comments
- [x] Semantic search baseline
- [x] Rerun comment gathering code to ignore mobile link bots
- [ ] Add transformer code to src (maybe can leverage sbert with no pretrained model?)
- [ ] Manually inspect some scraped websites and processed queries --> is the text clean enough??
- [ ] Do a full scrape, then get scores for all of the models
- [ ] Make test functions search over faiss instead of loading everything at once (only a problem with scale?)
- [ ] Use teacher to enforce negative samples with the margin (see [paper](https://arxiv.org/pdf/2010.02666.pdf), [code](https://github.com/UKPLab/sentence-transformers/blob/40af04ed70e16408f466faaa5243bee6f476b96e/examples/training/ms_marco/train_bi-encoder_mnrl.py#L149))
- [ ] Proper data loading, batch size for LSTM
- [ ] Add data parallel wrappers
- [ ] Use model files and place all hyper parameters in there


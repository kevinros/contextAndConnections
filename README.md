# Recommending Webpages to Online Discussion Forums
This project explores the feasibility of recommending webpages to online discussion forums. The instructions below detail how to download the data, set up the models, and reproduce the experimental results. 

## Getting the processed data
The data is publicly available for download from [Google Drive, approx. 3GB uncompressed](https://drive.google.com/drive/folders/1waiWBRrwuNh3tp1b8mzrCNyR-SJ8yiuy?usp=sharing). After download and extraction, you should have the following files:

1. ``webpages``, which contains the complete collected webpage corpus for our experiments. Each webpage is its own json file, and it contains two fields: ``id`` which is the same as the filename, uniquely identifying the webpage, and ``contents``, which is the scraped text.  
2. ``queries`` which should contain ``queries_{train,val,test}.tsv``,  ``url_file_map.pkl``, and ``relevance_scores.txt``. 

``queries_{train,val,test}.tsv``: contains the train/dev/test splits for the Full setting. The first item on each line is the query id, and the second item on each line is the comment chain, where each comment is separted by <C>. Note that only the Full setting is included, but creating the Last setting (keeping only the last comment) or the Proactive setting (removing the last comment) can be done easily. 

``url_file_map.pkl``: maps a url to the file name in ``webpages``. The dictionary map also contains status (if the URL was successful), the comment ids in the chains from the original Reddit data, and whether the URL is in the train/dev/test split.

``relevance_scores.txt``: trec style relevance judgements for each query id. Maps to a webpage id. 

For ease of use, you can then set up the following file structure:

```
src
|___data_2017-09
|___|____pyserini --> empty dir for Pyserini indexing
|___|____queries
|___|___|___queries_{train,val,test}.tsv
|___|___|___relevance_scores.txt
|___|____webpages
|___|___|___{webpage_id}.json --> all webpage files
|___|____RC_2017-09
```

## Getting data from scratch

### File structure descriptions and setup
See ``src/data_helpers/setup.txt`` for instructions. For ease of use, you can make the file structure look something like this:
```
src
|___data_2017-09
|___|____pyserini
|___|____queries
|___|____queries_onlylast
|___|____queries_removelast
|___|____webpages
|___|____RC_2017-09
```

### Processing data
Navigate to data helpers directory: ```cd src/data_helpers```

Follow directions in setup.txt to download Reddit data. Then go through the following:

Set up comment structure: ```python3 format_reddit_comments.py --out_path ../data_2017-09/ --raw_reddit_data ../data_2017-09/RC_2017-09```

This will create 
1. ```all_needed_comments.pkl``` A dictionary that maps comment id to raw comment body. Created based on the URLs found and their respective paths back to the root comment.
2. ```all_urls.pkl``` A dictionary that maps line number (corresponding to a comment, from the original reddit comment json file) to ```{'urls': [all urls in the comment], 'id': id of comment, 'chain': chain of ids from root to comment}```
3. ```valid_urls.pkl``` Same as ```all_urls.pkl```, but only for URLs that pass the selection criteria. 

Now, we can use the ids in the chains of ```valid_urls.pkl``` and ```all_needed_comments.pkl``` to reconstruct comment threads

Scrape URLs: ```python3 scrape_urls.py --min_length 2 --data_path ../data_2017-09/ --url_file_map ../data_2017-09/url_file_map.pkl```

This will create
1. A collection of scraped webpage text in the ```webpages``` directory (empty dir should have been created in setup.txt). Each webpage will have a unique file name and be formatted as a json (matching the expected input for Pyserini)
2. ```url_file_map.pkl``` A dictionary that maps a URL to ```{'filename': name of file in webpages, 'chains': list of comment chains ending with that url, 'status': whether or not the page was successfully scraped, 'split': train, val, or test}```

Next, we can build our queries and relevance score file: 
```python3 build_queries.py --data_path ../data_2017-09/ --out ../data_2017-09/queries/ --mode all```

```python3 build_queries.py --data_path ../data_2017-09/ --out ../data_2017-09/queries_onlylast/ --mode only_last```

```python3 build_queries.py --data_path ../data_2017-09/ --out ../data_2017-09/queries_removelast/ --mode remove_last```

This will create
1. A train/val/test tsv split of plaintext comment chains in the ```queries / queries_onlylast / queries_removelast``` directory (created in setup.txt). Each comment will be separated by a <C> tag.
2. ```relevance_scores.txt```. Each line maps a query id to a relevant webpage filename. 


To pre-encode the queries and webpages for the semantic baseline runs, see ``src/data_helpers/encode_dataset.py``:

## Training and evaluating the models

Assume you are in src directory

Run BM25 baseline: use ``src/bm25_baseline.ipynb``

Run Semantic baseline: use ``src/semantic_baseline.py``

Run Semantic finetune: use ``src/semantic_finetune.py``

Evaluate: see ``src/eval/evaluate.ipynb``


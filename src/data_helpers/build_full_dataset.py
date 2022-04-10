import argparse
import json
import random
import re
from sentence_transformers import SentenceTransformer, util
import pickle

import os


def process_bow_query(src: list, url: str, remove_last_comment=False) -> str:
    '''
    bow processing code
    '''
    if remove_last_comment:
        src = src[:-1]
    else:
        src[-1] = re.sub(re.escape(url), ' ', src[-1])

    processed_src = " ".join(src)
    if len(processed_src) > 1024:
        processed_src = processed_src[-1024:]
    processed_src = re.sub('\n', '', processed_src)
    return processed_src

def process_encode_query(src: list, url: str, remove_last_comment=False) -> list:
    if remove_last_comment:
        src = src[:-1]
    else:
        src[-1] = re.sub(re.escape(url), ' ', src[-1])

    processed_src = [re.sub('\n', '', x) for x in src]
    return processed_src




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--formatted', help="path to formatted reddit comments json")
    parser.add_argument('--webpages', help="path to scraped web pages json")

    parser.add_argument('--bow', help='make bag of words data set')
    parser.add_argument('--encode', help='make encoded data set')

    parser.add_argument('--out', help='dir path to save data sets')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"

    # example usage
    # python3 build_full_dataset.py --formatted ../data/RC_2009-05_formatted.json --webpages ../data/RC_2009-05_webpages.json --bow yes --encode yes --out ../data/


    random.seed(100)

    formatted = json.load(open(args.formatted, 'r'))
    webpages = json.load(open(args.webpages, 'r'))

    encoded_queries = []
    encoded_websites = {}
    bow_queries = []
    bow_websites = []

    relevance_scores = []

    random.shuffle(formatted)

    sbert_model = SentenceTransformer('msmarco-distilbert-cos-v5')

    # maps a given url to the webpages index so that the relevance scores can be built
    # using the url of a query
    url_idx = {} 
    for i,url in enumerate(webpages):
        url_idx[url] = i

    # build the corpus
    for i,webpage in enumerate(webpages):

        if args.bow:
            bow_websites.append({"id": i, "contents": webpages[webpage]})
        if args.encode:
            encoded_websites[i] = sbert_model.encode(webpages[webpage], convert_to_tensor=True)
                
        if i % 100 == 0:
            print('Webpages processed: ' + str(i) + ' out of ' + str(len(webpages)))

    print(len(bow_websites), ' total processed bow websites')
    print(len(encoded_websites), ' total encoded websites')


    # build the queries
    query_idx = 0
    for thread in formatted:
        url = thread['url']
        
        # unable to scrape it
        if url not in webpages:
            continue

        if args.bow:
            processed_query = process_bow_query(thread['full_context'], url)
            bow_queries.append(str(query_idx) + '\t' + processed_query)
        
        if args.encode:
            processed_query = process_encode_query(thread['full_context'], url)
            encoded_queries.append({"query_id": query_idx, 'encoding': sbert_model.encode(processed_query, convert_to_tensor=True)})
    
        # build the relevance scores https://trec.nist.gov/data/qrels_eng/
        relevance_scores.append(str(query_idx) + ' 0 ' + str(url_idx[url]) + ' 1')
        query_idx += 1

        if query_idx % 100 == 0:
            print('Queries processed: ' + str(query_idx) + ' out of ' + str(len(formatted)))

    print(len(bow_queries), ' total processed bow queries')
    print(len(encoded_queries), ' total encoded queries')

    train_val_split = 0.8
    val_test_split = 0.9

    if args.bow:
        # need to put bow_websites in its own directory because of how pyserini indexes it
        with open(args.out + '/websites/bow_websites.jsonl', 'w') as f:
            for doc in bow_websites:
                f.write(json.dumps(doc) + '\n')


        total_len = len(bow_queries)
        src_train = bow_queries[:int(total_len * train_val_split)]
        src_val = bow_queries[int(total_len * train_val_split): int(total_len * val_test_split)]
        src_test = bow_queries[int(total_len * val_test_split):]

        with open(args.out + 'bow_queries_train.tsv', 'w') as f:
            for doc in src_train:
                f.write(doc + '\n')
        with open(args.out + 'bow_queries_dev.tsv', 'w') as f:
            for doc in src_val:
                f.write(doc + '\n')
        with open(args.out + 'bow_queries_test.tsv', 'w') as f:
            for doc in src_test:
                f.write(doc + '\n')


    if args.encode:
        total_len = len(encoded_queries)
        src_train = encoded_queries[:int(total_len * train_val_split)]
        src_val = encoded_queries[int(total_len * train_val_split): int(total_len * val_test_split)]
        src_test = encoded_queries[int(total_len * val_test_split):]

        pickle.dump(src_train, open(args.out + 'encoded_queries_train.pkl', 'wb'))
        pickle.dump(src_val, open(args.out + 'encoded_queries_dev.pkl', 'wb'))
        pickle.dump(src_test, open(args.out + 'encoded_queries_test.pkl', 'wb'))


        pickle.dump(encoded_websites, open(args.out + 'encoded_websites.pkl', 'wb'))

    with open(args.out + 'relevance_scores.txt', 'w') as f:
        for rs in relevance_scores:
            f.write(rs + '\n')



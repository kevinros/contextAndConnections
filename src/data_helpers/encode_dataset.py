import argparse
import json
import re
from sentence_transformers import SentenceTransformer
import pickle
import os

# example run 
# python3 encode_dataset.py --query_path ../data_2017-09/queries/queries_train.tsv --out ../data_2017-09/encoded_queries/queries_train.pkl
# python3 encode_dataset.py --query_path ../data_2017-09/queries/queries_val.tsv --out ../data_2017-09/encoded_queries/queries_val.pkl
# python3 encode_dataset.py --webpages_path ../data_2017-09/webpages/ --out ../data_2017-09/encoded_webpages/webpages.pkl

def process_query(src: list, remove_last_comment=False) -> list:
    src = src.split('<C>')
    if remove_last_comment:
        src = src[:-1]

    processed_src = [re.sub('\n', '', x) for x in src]
    return processed_src

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help="path to query tsv file")
    parser.add_argument('--webpages_path', help="path webpage directory")
    parser.add_argument('--out', help='path to save encodings')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
    sbert_model = SentenceTransformer('msmarco-distilbert-dot-v5')


    if args.query_path:
        encoded_queries = []
        with open(args.query_path, 'r') as f:
            for i,line in enumerate(f):
                split_line = line.split('\t')
                query_id = split_line[0]
                query = " ".join(split_line[1:])
                comments = process_query(query)
                encoded_queries.append({"query_id": query_id, 'encoding': sbert_model.encode(comments, convert_to_tensor=True)})

                if i % 100 == 0:
                    print('Queries processed: ' + str(i))

        print('Total encoded queries :', len(encoded_queries))
        pickle.dump(encoded_queries, open(args.out, 'wb'))


    elif args.webpages_path:
        encoded_webpages = {}
        for i,file in enumerate(os.listdir(args.webpages_path)):
            webpage = json.load(open(args.webpages_path + file, 'r'))
            encoded_webpage = sbert_model.encode(webpage['contents'], convert_to_tensor=True)
            encoded_webpages[webpage['id']] = encoded_webpage

            if i % 100 == 0:
                print('Webpages processed: ' + str(i))

        print('Total encoded webpages :', len(encoded_webpages))
        pickle.dump(encoded_webpages, open(args.out, 'wb'))
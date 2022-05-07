import argparse
import json
import re
from sentence_transformers import SentenceTransformer
import pickle
import os
import hnswlib

# example run 
# python3 encode_dataset.py --query_path ../data_2017-09/queries/queries_train.tsv --out ../data_2017-09/encoded_queries/queries_train.pkl
# python3 encode_dataset.py --query_path ../data_2017-09/queries/queries_val.tsv --out ../data_2017-09/encoded_queries/queries_val.pkl
# python3 encode_dataset.py --webpages_path ../data_2017-09/webpages/ --out ../data_2017-09/encoded_webpages/

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
    parser.add_argument('-model', help="path to model")
    parser.add_argument('--out', help='path to save encodings')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"

    if args.model:
        sbert_model = SentenceTransformer(args.model)
        index_name = 'webpages_' + args.model_name
    else:
        sbert_model = SentenceTransformer('msmarco-distilbert-dot-v5')
        index_name = 'webpages_baseline'



    if args.query_path:
        encoded_queries = []
        with open(args.query_path, 'r') as f:
            for i,line in enumerate(f):
                split_line = line.split('\t')
                query_id = split_line[0]
                query = " ".join(split_line[1:])
                comments = process_query(query)
                encoded_queries.append({"query_id": query_id, 'encoding': sbert_model.encode(comments, convert_to_tensor=True)})

                if i % 1000 == 0:
                    print('Queries processed: ' + str(i))

        print('Total encoded queries :', len(encoded_queries))
        pickle.dump(encoded_queries, open(args.out, 'wb'))


    elif args.webpages_path:
        encoded_webpages = []
        encoded_webpage_ints = []
        int_id_map = {}
        for i,file in enumerate(os.listdir(args.webpages_path)):
            webpage = json.load(open(args.webpages_path + file, 'r'))
            encoded_webpage = sbert_model.encode(webpage['contents'])

            encoded_webpages.append(encoded_webpage)
            encoded_webpage_ints.append(i)
            int_id_map[i] = webpage['id']

            if i % 1000 == 0:
                print('Webpages processed: ' + str(i))

        print('Total encoded webpages :', len(encoded_webpages))
        index = hnswlib.Index(space='ip', dim=768)
        index.init_index(max_elements = len(encoded_webpages), ef_construction = 200, M = 16)
        index.add_items(encoded_webpages, encoded_webpage_ints)
        print('Index created')
        index.save_index(args.out + index_name + '.bin')
        pickle.dump(int_id_map, open(args.out + 'int_id_map_' + index_name + '.pkl', 'wb'))
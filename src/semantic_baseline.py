import argparse
import pickle
from sentence_transformers import util
import torch
import hnswlib

# example usage
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_baseline.bin --index_map data_2017-09/encoded_webpages/int_id_map_webpages_baseline.pkl --queries data_2017-09/encoded_queries/queries_val.pkl --out out/semantic_runs/run.val.txt

def semantic_baseline(index, int_id_map, queries, k=10):
    run = []

    for i,query in enumerate(queries):

        query_id = query['query_id']

        # get the last comment (with the missing URL)
        x = query['encoding'][-1].cpu()

        labels, distances = index.knn_query(x, k)

        for j, (idx, score) in enumerate(zip(labels[0], distances[0])):
            # 0 Q0 0 1 193.457108 Anserini
            run.append(' '.join([str(query_id), 'Q0', str(int_id_map[idx.item()]), str(j), str(score.item()), 'Semantic Search']))
    return run


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--relevance_scores', help="path to relevance scores")
    parser.add_argument('--index', help="path to website index")
    parser.add_argument('--index_map', help="path to index id - webpage id map")
    parser.add_argument('--queries', help="path to encoded queries")
    parser.add_argument('--out', help="path to output directory to save run")

    args = parser.parse_args()

    queries = pickle.load(open(args.queries, 'rb'))
    print('Loaded queries')

    index = hnswlib.Index(space='ip', dim=768)
    index.set_ef(1000)
    index.load_index(args.index)

    int_id_map = pickle.load(open(args.index_map, 'rb'))

    print('Index set up')

    run = semantic_baseline(index, int_id_map, queries)


    with open(args.out, 'w') as f:
        for line in run:
            f.write(line + '\n')
    


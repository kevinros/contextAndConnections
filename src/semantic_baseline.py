import argparse
import pickle
from sentence_transformers import util
import torch
import hnswlib
import os

# example usage

# for semantic finetune, make sure to change data to correspond to model
# for validation data
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_2022-06-01_12-04-50.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_2022-06-01_12-04-50.pkl --queries data_2017-09/queries/queries_val_2022-06-01_12-04-50.pkl --out out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries-2022-06-01_12-04-50/eval/run.val_full.txt
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_2022-06-02_21-24-49.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_2022-06-02_21-24-49.pkl --queries data_2017-09/queries_onlylast/queries_val_2022-06-02_21-24-49.pkl --out out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries_onlylast-2022-06-02_21-24-49/eval/run.val_onlylast.txt
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_2022-05-22_18-44-28.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_2022-05-22_18-44-28.pkl --queries data_2017-09/queries_removelast/queries_val_2022-05-22_18-44-28.pkl --out out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries_removelast-2022-05-22_18-44-28/eval/run.val_removelast.txt
# for test data
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_2022-06-01_12-04-50.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_2022-06-01_12-04-50.pkl --queries data_2017-09/queries/queries_test_2022-06-01_12-04-50.pkl --out out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries-2022-06-01_12-04-50/eval/v2_run.test_full.txt
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_2022-06-02_21-24-49.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_2022-06-02_21-24-49.pkl --queries data_2017-09/queries_onlylast/queries_test_2022-06-02_21-24-49.pkl --out out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries_onlylast-2022-06-02_21-24-49/eval/v2_run.test_onlylast.txt
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_2022-05-22_18-44-28.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_2022-05-22_18-44-28.pkl --queries data_2017-09/queries_removelast/queries_test_2022-05-22_18-44-28.pkl --out out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries_removelast-2022-05-22_18-44-28/eval/run.test_removelast.txt



# for semantic baseline
# for validation data
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_msmarco-distilbert-cos-v5.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_msmarco-distilbert-cos-v5.pkl --queries data_2017-09/queries/queries_val_cos.pkl --out out/semantic_runs/v2_run.val_full_msmarco-distilbert-cos-v5.txt
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_msmarco-distilbert-cos-v5.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_msmarco-distilbert-cos-v5.pkl --queries data_2017-09/queries_onlylast/queries_val_cos.pkl --out out/semantic_runs/v2_run.val_onlylast_msmarco-distilbert-cos-v5.txt
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_msmarco-distilbert-cos-v5.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_msmarco-distilbert-cos-v5.pkl --queries data_2017-09/queries_removelast/queries_val_cos.pkl --out out/semantic_runs/run.val_removelast_msmarco-distilbert-cos-v5.txt
# for test data
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_msmarco-distilbert-cos-v5.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_msmarco-distilbert-cos-v5.pkl --queries data_2017-09/queries/queries_test_cos.pkl --out out/semantic_runs/v2_run.test_full_msmarco-distilbert-cos-v5.txt
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_msmarco-distilbert-cos-v5.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_msmarco-distilbert-cos-v5.pkl --queries data_2017-09/queries_onlylast/queries_test_cos.pkl --out out/semantic_runs/v2_run.test_onlylast_msmarco-distilbert-cos-v5.txt
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_msmarco-distilbert-cos-v5.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_msmarco-distilbert-cos-v5.pkl --queries data_2017-09/queries_removelast/queries_test_cos.pkl --out out/semantic_runs/run.test_removelast_msmarco-distilbert-cos-v5.txt


# not used
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_baseline.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_baseline.pkl --queries data_2017-09/queries/queries_val.pkl --out out/semantic_runs/run.val_full_cos.txt
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_baseline.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_baseline.pkl --queries data_2017-09/queries_onlylast/queries_val.pkl --out out/semantic_runs/run.val_onlylast_cos.txt
# python3 semantic_baseline.py --index data_2017-09/encoded_webpages/webpages_baseline.pkl --index_map data_2017-09/encoded_webpages/int_id_map_webpages_baseline.pkl --queries data_2017-09/queries_removelast/queries_val.pkl --out out/semantic_runs/run.val_removelast_cos.txt

def semantic_baseline(index, int_id_map, queries, k=10):
    run = []

    for i,query in enumerate(queries):

        query_id = query['query_id']

        x = query['encoding'].cpu()

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

    os.environ['CUDA_VISIBLE_DEVICES'] = "2"


    queries = pickle.load(open(args.queries, 'rb'))

    query_ids = [x['query_id'] for x in queries]
    queries = [x['encoding'] for x in queries]

    print('Loaded queries')

    int_id_map = pickle.load(open(args.index_map, 'rb'))

    corpus = pickle.load(open(args.index, 'rb'))

    hits = util.semantic_search(queries, corpus, score_function=util.dot_score)
    #hits = util.semantic_search(queries, corpus, score_function=util.cos_sim)


    run = []
    for i,query in enumerate(hits):
        for j, result in enumerate(query):
            # 0 Q0 0 1 193.457108 Anserini
            run.append(' '.join([str(query_ids[i]), 'Q0', str(int_id_map[result['corpus_id']]), str(j), str(result['score']), 'Semantic Search']))
    print('Index set up')

    with open(args.out, 'w') as f:
        for line in run:
            f.write(line + '\n')
    


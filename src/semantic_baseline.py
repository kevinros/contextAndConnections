import argparse
import pickle
from sentence_transformers import util
import torch

# example usage
# python3 semantic_baseline.py --corpus data_2017-09/encoded_webpages/webpages.pkl --queries data_2017-09/encoded_queries/queries_train.pkl --out out/semantic_runs/

def semantic_baseline(corpus, queries, k=10):
    # TODO: combine with LSTM test method
    run = []
    # so that we can do cosine scores and recover the actual id of the webpage
    just_corpus_encodings = [torch.unsqueeze(corpus[x],dim=0) for x in corpus]
    just_corpus_encodings = torch.cat(just_corpus_encodings)

    webpage_id_map = {}
    for i,webpage_id in enumerate(corpus):
        webpage_id_map[i] = webpage_id


    for i,query in enumerate(queries):

        query_id = query['query_id']

        # get the last comment (with the missing URL)
        x = query['encoding'][-1] 

        scores = util.dot_score(x, just_corpus_encodings)[0]
        top_results = torch.topk(scores, k=k)
        for j, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
            # 0 Q0 0 1 193.457108 Anserini
            run.append(' '.join([str(query_id), 'Q0', str(webpage_id_map[idx.item()]), str(j), str(score.item()), 'Semantic Search']))
    return run


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--relevance_scores', help="path to relevance scores")
    parser.add_argument('--corpus', help="path to website encodings")
    parser.add_argument('--queries', help="path to encoded queries")
    parser.add_argument('--out', help="path to output directory to save run")

    args = parser.parse_args()

    queries = pickle.load(open(args.queries, 'rb'))

    corpus = pickle.load(open(args.corpus, 'rb'))

    run = semantic_baseline(corpus, queries)


    with open(args.out + 'run.train.txt', 'w') as f:
        for line in run:
            f.write(line + '\n')
    


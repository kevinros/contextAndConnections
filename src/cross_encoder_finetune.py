import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, evaluation, losses, InputExample, util
import logging
from datetime import datetime
import os
from torch.utils.data import Dataset
import random
import argparse
import torch
from typing import List
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
from eval import eval
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator




# example usage

# to train:
# python3 cross_encoder_finetune.py --corpus data_2017-09/webpages/ --queries data_2017-09/queries/ --model_name cross-encoder/ms-marco-MiniLM-L-4-v2 --neg_sample_run_path out/bm25_runs/v2_run.train_8_0.99.txt



def load_queries(path, corpus, relevance_scores, neg_sample_run_path=None, num_neg=1, type='train'):
    queries = {}

    if type == "train":
        neg_sample_run = eval.load_run(neg_sample_run_path)
    
    rel_score_keys = list(relevance_scores.keys())
    num_options = len(rel_score_keys)-1


    with open(path, 'r') as f:
        for i,line in enumerate(f):
            split_line = line.split('\t')
            query_id = split_line[0]
            query = " ".join(split_line[1:])
            queries[query_id] = {'query': query, 'positive': [], 'negative': []}

            webpage = corpus[relevance_scores[query_id]]

            # reverse for truncation, so that the most recent comments aren't removed
            query = " <C> ".join(query.split('<C>')[::-1])

            #queries.append(InputExample(texts=[query, webpage], label=1))
            queries[query_id]['positive'].append(webpage)
            #if type == "val" or type == "test":
            #    continue

            neg_pids = []

            # there is an unhandled case where if the query is empty, then it doesn't show up in bm25
            # unclear how to handle, so for now, if that happens, we'll just randomly sample
            # second condition for when there is only one result returned by bm25
            if True:#query_id not in neg_sample_run or len(neg_sample_run[query_id]) == 1:
                for _ in range(0, num_neg):
                    rand_idx = random.randint(0,num_options)
                    neg_webpage = corpus[relevance_scores[rel_score_keys[rand_idx]]]
                    #queries.append(InputExample(texts=[query, neg_webpage], label=0))
                    queries[query_id]['negative'].append(neg_webpage)

            else:
                # use another run, automatically takes the highest not equal to ground truth
                # note max num_neg is 10 currently
                for i,returned_doc_id in enumerate(neg_sample_run[query_id]):
                    if returned_doc_id == relevance_scores[query_id]: continue
                    if len(neg_pids) == num_neg: break
                    neg_webpage = corpus[returned_doc_id]
                    queries.append(InputExample(texts=[query, neg_webpage], label=0))
    return queries

def load_webpages(path):
    webpages = {}
    for file in os.listdir(path):
        webpage = json.load(open(path + file, 'r'))
        webpages[webpage['id']] = webpage['contents']
    return webpages


if __name__ == '__main__':

    random.seed(100)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", default=20, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--model_name", default="msmarco-distilbert-dot-v5")
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--warmup_steps", default=300, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_negs_per_system", default=50, type=int)
    parser.add_argument("--use_pre_trained_model", default=True, action="store_true")
    parser.add_argument("--use_all_queries", default=False, action="store_true")
    parser.add_argument("--neg_sample_run_path", required=True)

    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)

    

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"


    model_name = args.model_name
    train_batch_size = args.train_batch_size         
    max_seq_length = args.max_seq_length            
    num_negs_per_system = args.num_negs_per_system         
    num_epochs = args.epochs
    query_path = args.queries
    corpus_path = args.corpus

    typ = query_path.split('/')[-2]

    if args.use_pre_trained_model:
        logging.info("use pretrained SBERT model")
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-4-v2', num_labels=1)
        model.max_seq_length = max_seq_length
    else:
        logging.info("Create new SBERT model")
        exit()
        #word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        #pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
        #model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    model_save_path = 'out/crossencoder_finetune_runs/train_crossencoder-mnrl-{}-{}-{}'.format(model_name.replace("/", "-"), typ, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(model_save_path)

    relevance_scores = eval.load_rel_scores(query_path + 'relevance_scores.txt')
    print('Relevance scores loaded')

    print('Using ', args.neg_sample_run_path, ' for negative samples')

    corpus = load_webpages(corpus_path)

    print('Corpus loaded')

    train_queries = load_queries(query_path + 'queries_train.tsv', corpus, relevance_scores, num_neg = num_negs_per_system, neg_sample_run_path=args.neg_sample_run_path)
    val_queries = load_queries(query_path + 'queries_val.tsv', corpus, relevance_scores, num_neg = num_negs_per_system, neg_sample_run_path=args.neg_sample_run_path, type="val")

    print('Queries loaded')

    print(len(train_queries), ' training queries')
    print(len(val_queries), ' validation queries')


    #train_dataset = WebpageDataset(train_queries, corpus=corpus)
    #train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_dataloader = DataLoader(train_queries, shuffle=True, batch_size=train_batch_size)

    #train_loss = losses.MultipleNegativesRankingLoss(model=model, similarity_fct=util.cos_sim) #util.dot_score

    #val_corpus = {}
    #for query in val_queries:
    #    query_doc = relevance_scores[query]
    #    val_corpus[query_doc] = corpus[query_doc]

    #evaluator = evaluation.InformationRetrievalEvaluator(val_queries, corpus, relevance_scores, corpus_chunk_size=1000)#, main_score_function="dot_score")
    #evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(val_queries, name='val')
    evaluator = CERerankingEvaluator(val_queries, name='train-eval')


    print('Beginning to train')
    model.fit(train_dataloader=train_dataloader,
          epochs=num_epochs,
          evaluator=evaluator,
          evaluation_steps=3500,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          optimizer_params = {'lr': args.lr},
          output_path=model_save_path,
    )
    


import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import os
from torch.utils.data import Dataset
import random
import argparse
import torch
from typing import List
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction

# example usage
# python3 semantic_finetune.py --corpus data_2017-09/webpages/ --queries data_2017-09/queries/


def load_rel_scores(path):
    query_webpage_map = {}
    with open(path, 'r') as relevance_scores:
        for line in relevance_scores:
            split_line = line.split()
            query_webpage_map[split_line[0]] = split_line[2]
    return query_webpage_map


def load_queries(path, relevance_scores, num_neg=5, type='train'):
    queries = {}
    rel_score_keys = list(relevance_scores.keys())
    num_options = len(rel_score_keys)-1
    with open(path, 'r') as f:
        for i,line in enumerate(f):
            split_line = line.split('\t')
            query_id = split_line[0]
            query = " ".join(split_line[1:])

            if type == "val" or type == "test":
                queries[query_id] = query
                if i > 1000: break # TODO: remove this once the training process is fast enough to handle the full val set
                continue

            # The original implementation chooses "hard samples". We could probably do something like that, i.e., using the BM25 run
            # For now, it's just random
            neg_pids = []
            for _ in range(0, num_neg):
                rand_idx = random.randint(0,num_options)
                neg_pids.append(relevance_scores[rel_score_keys[rand_idx]])
            queries[query_id] = {'qid': query_id, 'query': query, 'pos': [relevance_scores[query_id]], 'neg': neg_pids}
    return queries

def load_webpages(path):
    webpages = {}
    for file in os.listdir(path):
        webpage = json.load(open(path + file, 'r'))
        webpages[webpage['id']] = webpage['contents']
    return webpages


class WebpageDataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query['pos'].append(pos_id)

        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)



if __name__ == '__main__':

    random.seed(100)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", default=5, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--model_name", default="msmarco-distilbert-dot-v5")
    parser.add_argument("--max_passages", default=0, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_negs_per_system", default=1, type=int)
    parser.add_argument("--use_pre_trained_model", default=True, action="store_true")
    parser.add_argument("--use_all_queries", default=False, action="store_true")
    parser.add_argument("--ce_score_margin", default=3.0, type=float)

    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)

    

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"


    model_name = args.model_name
    train_batch_size = args.train_batch_size         
    max_seq_length = args.max_seq_length            
    ce_score_margin = args.ce_score_margin            
    num_negs_per_system = args.num_negs_per_system         
    num_epochs = args.epochs


    query_path = args.queries
    corpus_path = args.corpus

    if args.use_pre_trained_model:
        logging.info("use pretrained SBERT model")
        model = SentenceTransformer(model_name)
        model.max_seq_length = max_seq_length
    else:
        logging.info("Create new SBERT model")
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    #model = torch.nn.DataParallel(model)


    model_save_path = 'out/semantic_finetune_runs/train_bi-encoder-mnrl-{}-margin_{:.1f}-{}'.format(model_name.replace("/", "-"), ce_score_margin, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


    relevance_scores = load_rel_scores(query_path + 'relevance_scores.txt')
    print('Relevance scores loaded')

    train_queries = load_queries(query_path + 'queries_train.tsv', relevance_scores, num_negs_per_system)
    val_queries = load_queries(query_path + 'queries_val.tsv', relevance_scores, type="val")

    print('Queries loaded')

    print(len(train_queries), ' training queries')
    print(len(val_queries), ' validation queries')


    corpus = load_webpages(corpus_path)

    print('Corpus loaded')

    train_dataset = WebpageDataset(train_queries, corpus=corpus)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    val_corpus = {}
    for query in val_queries:
        query_doc = relevance_scores[query]
        val_corpus[query_doc] = corpus[query_doc]

    evaluator = evaluation.InformationRetrievalEvaluator(val_queries, val_corpus, relevance_scores, corpus_chunk_size=100)
    # going to be super slow because it needs to encode the ENTIRE corpus
    # for now, just compute embedding similarity
    #val_query_src = [val_queries[x] for x in val_queries]
    #val_query_tgt = [corpus[relevance_scores[x]] for x in val_queries]
    #evaluator = evaluation.EmbeddingSimilarityEvaluator(val_query_src, val_query_tgt)

    

    print('Beginning to train')
    model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          evaluator=evaluator,
          evaluation_steps=500,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          optimizer_params = {'lr': args.lr},
          output_path=model_save_path,
          )
    


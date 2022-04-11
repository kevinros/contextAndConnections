import argparse
import os
import pickle
from sentence_transformers import util


from models import lstm
import torch


def semantic_baseline(corpus, X_test, k=10):
    # TODO: combine with LSTM test method
    run = []
    # so that we can do cosine scores and recover the actual id of the webpage
    just_corpus_encodings = [torch.unsqueeze(corpus[x],dim=0) for x in corpus]
    just_corpus_encodings = torch.cat(just_corpus_encodings)

    webpage_id_map = {}
    for i,webpage_id in enumerate(corpus):
        webpage_id_map[i] = webpage_id


    for i,query in enumerate(X_test):

        query_id = query['query_id']

        # get the last comment (with the missing URL)
        x = query['encoding'][-1] 

           
        cos_scores = util.cos_sim(x, just_corpus_encodings)[0]
        top_results = torch.topk(cos_scores, k=k)
        for j, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
            # 0 Q0 0 1 193.457108 Anserini
            run.append(' '.join([str(query_id), 'Q0', str(webpage_id_map[idx.item()]), str(j), str(score.item()), 'Semantic Search']))
    return run


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="name of the neural model")
    parser.add_argument('--relevance_scores', help="path to relevance scores")
    parser.add_argument('--corpus', help="path to website encodings")

    parser.add_argument('--src_train', help="path to queries (train)")
    parser.add_argument('--src_dev', help="path to queries (dev)")
    parser.add_argument('--src_test', help="path to queries (test)")

    parser.add_argument('--out', help="path to output directory")

    args = parser.parse_args()

    # example usage
    # python3 neural.py --model lstm --relevance_scores data/relevance_scores.txt --corpus data/encoded_websites.pkl --src_train data/encoded_queries_train.pkl --src_dev data/encoded_queries_dev.pkl --src_test data/encoded_queries_test.pkl --out out/lstm_runs/
    # python3 neural.py --model semantic_baseline --relevance_scores data/relevance_scores.txt --corpus data/encoded_websites.pkl --src_train data/encoded_queries_train.pkl --src_dev data/encoded_queries_dev.pkl --src_test data/encoded_queries_test.pkl --out out/semantic_runs/

    os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"

    src_train = pickle.load(open(args.src_train, 'rb'))
    src_dev = pickle.load(open(args.src_dev, 'rb'))
    src_test = pickle.load(open(args.src_test, 'rb'))

    corpus = pickle.load(open(args.corpus, 'rb'))

    # need to map queries to websites
    relevance_scores = open(args.relevance_scores, 'r')
    query_webpage_map = {}
    for line in relevance_scores:
        split_line = line.split()
        query_webpage_map[int(split_line[0])] = int(split_line[2])


    if args.model == "lstm":
        input_size = 768
        hidden_size = 768
        num_layers = 2
        learning_rate = 1e-4
        warm_up_rate = 0.1
        epochs = 10


        model = lstm.URLSTM(input_size, hidden_size, num_layers)
        model.to('cuda:0')

        loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


        trainer = lstm.LSTMTrainer(model, loss_fn, optimizer, corpus, query_webpage_map)

        for epoch in range(epochs):
            print('Epoch: ', epoch)
            if warm_up_rate < 1:
                optimizer.param_groups[0]['lr'] = learning_rate * warm_up_rate
                warm_up_rate *= 2

            loss_train = trainer.train(src_train)
            loss_dev = trainer.dev(src_dev)
            print('Total train loss: ', loss_train)
            print('Total dev loss: ', loss_dev)
        run = trainer.test(src_dev)

        torch.save(model, args.out + 'model')


    if args.model == "semantic_baseline":
        run = semantic_baseline(corpus, src_dev)


    with open(args.out + 'run.dev.txt', 'w') as f:
        for line in run:
            f.write(line + '\n')
    


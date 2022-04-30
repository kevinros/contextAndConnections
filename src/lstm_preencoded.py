import argparse
import os
import pickle
from sentence_transformers import util
from torch import nn, Tensor

from models import lstm
import torch


# example usage
# python3 lstm_preencoded.py --relevance_scores data_2017-09/queries/relevance_scores.txt --corpus data_2017-09/encoded_webpages/webpages.pkl --queries_train data_2017-09/encoded_queries/queries_train.pkl --queries_val data_2017-09/encoded_queries/queries_val.pkl --out out/lstm_preencoded_runs/



class MarginMSELoss(nn.Module):
    def __init__(self, similarity_fct=util.pairwise_dot_score):
        super(MarginMSELoss, self).__init__()
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()
    def forward(self, query, positive, negative, margin):
        scores_pos = self.similarity_fct(query, positive)
        scores_neg = self.similarity_fct(query, negative)
        diff = scores_pos - scores_neg
        return self.loss_fct(diff, margin)




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--relevance_scores', help="path to relevance scores")
    parser.add_argument('--corpus', help="path to website encodings")
    parser.add_argument('--queries_train', help="path to queries (train)")
    parser.add_argument('--queries_val', help="path to queries (validation)")
    parser.add_argument('--out', help="path to output directory")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"

    queries_train = pickle.load(open(args.queries_train, 'rb'))
    queries_val= pickle.load(open(args.queries_val, 'rb'))

    corpus = pickle.load(open(args.corpus, 'rb'))

    # need to map queries to websites
    relevance_scores = open(args.relevance_scores, 'r')
    query_webpage_map = {}
    for line in relevance_scores:
        split_line = line.split()
        query_webpage_map[split_line[0]] = split_line[2]


    input_size = 768
    hidden_size = 768
    num_layers = 2
    learning_rate = 1e-4
    warm_up_rate = 0.1
    epochs = 100


    model = lstm.URLSTM(input_size, hidden_size, num_layers)
    model.to('cuda:0')

    loss_fn = MarginMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    trainer = lstm.LSTMTrainer(model, loss_fn, optimizer, corpus, query_webpage_map)

    for epoch in range(epochs):
        print('Epoch: ', epoch)
        if warm_up_rate < 1:
            optimizer.param_groups[0]['lr'] = learning_rate * warm_up_rate
            warm_up_rate *= 2

        loss_train = trainer.train(queries_train)
        loss_val = trainer.val(queries_val)
        print('Total train loss: ', loss_train)
        print('Total validation loss: ', loss_val)
    run = trainer.test(queries_val)

    torch.save(model, args.out + 'model')

    with open(args.out + 'run.val.txt', 'w') as f:
        for line in run:
            f.write(line + '\n')
    

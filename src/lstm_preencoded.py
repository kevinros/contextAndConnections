import argparse
import os
import pickle
from sentence_transformers import util
from torch import nn, Tensor
from models import lstm
import torch

# example usage
# python3 lstm_preencoded.py --relevance_scores data_2017-09/queries/relevance_scores.txt --corpus data_2017-09/encoded_webpages/webpages.pkl --queries_train data_2017-09/queries/queries_train.pkl --queries_val data_2017-09/queries/queries_val.pkl --out out/lstm_runs/ --index_map data_2017-09/encoded_webpages/int_id_map_webpages.pkl --setting full

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
    parser.add_argument('--index_map', help="path to index - webpage id map")
    parser.add_argument('--queries_train', help="path to queries (train)")
    parser.add_argument('--queries_val', help="path to queries (validation)")
    parser.add_argument('--out', help="path to output directory")
    parser.add_argument('--setting', help="full or proactive")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"

    queries_train = pickle.load(open(args.queries_train, 'rb'))
    queries_val = pickle.load(open(args.queries_val, 'rb'))
    corpus = pickle.load(open(args.corpus, 'rb'))
    int_id_map = pickle.load(open(args.index_map, 'rb'))
    id_int_map = {value:key for key,value in int_id_map.items()}
    
    # generate ground truth output data
    just_corpus_encodings = torch.unsqueeze(corpus[0], dim=0)
    for x in corpus[1:]:
        just_corpus_encodings = torch.cat((just_corpus_encodings, torch.unsqueeze(x, dim=0)), 0)

    # need to map queries to websites
    relevance_scores = open(args.relevance_scores, 'r')
    query_webpage_map = {}
    for line in relevance_scores:
        split_line = line.split()
        query_webpage_map[split_line[0]] = split_line[2]

    # parameters
    input_size = 768
    hidden_size = 768
    num_layers = 1
    learning_rate = 1e-4
    warm_up_rate = 0.1 
    epochs = 20
    marg = 0
    batch_size = 5
    num_of_negative_samples = 20

    model = lstm.URLSTM(input_size, hidden_size, num_layers)
    model.to('cuda:0')

    loss_fn = nn.CosineEmbeddingLoss(margin=marg)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = lstm.LSTMTrainer(model, loss_fn, optimizer, corpus, query_webpage_map, 
                                id_int_map, int_id_map, relevance_scores, just_corpus_encodings, args.setting)

    for epoch in range(epochs):
        print('Epoch: ', epoch)
        if warm_up_rate < 1:
            optimizer.param_groups[0]['lr'] = learning_rate * warm_up_rate
            warm_up_rate *= 2

        loss_train = trainer.train(queries_train, batch_size, num_of_negative_samples)
        loss_val = trainer.val(queries_val, epoch, epochs, num_layers, 
                                batch_size, num_of_negative_samples, marg, learning_rate, warm_up_rate)

        print('Total train loss: ', loss_train)
        print('Total validation loss: ', loss_val)

    run = trainer.test(queries_val)
    torch.save(model, args.out + 'model')

    with open(args.out + 'run.val' + str(epochs) + '_' + str(num_layers) + '_' + str(warm_up_rate) + '_' + str(learning_rate) + '_'
                       + str(marg) + '_' + str(batch_size) + '_' + str(num_of_negative_samples) + '.txt', 'w') as f:
        for line in run:
            f.write(line + '\n')
    
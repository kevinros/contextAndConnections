from torch import nn, zeros
import torch
import random
from sentence_transformers import util


class URLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(URLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        return output, state

    def init_state(self):
        return (zeros(self.num_layers, 1, self.input_size),
                zeros(self.num_layers, 1, self.input_size))


class LSTMTrainer():
    def __init__(self, model, loss_fn, optimizer, corpus, query_webpage_map):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.corpus = corpus
        self.query_webpage_map = query_webpage_map

        self.neg_samples = 5


    def train(self, X_train):
        total_loss = 0
        self.model.train()
        for i,query in enumerate(X_train):
            query_id = query['query_id']
            x = query['encoding']
            y = self.corpus[self.query_webpage_map[query_id]]

            state_h, state_c = self.model.init_state()
            state_h = state_h.to('cuda:0')
            state_c = state_c.to('cuda:0')

            pred, (state_h, state_c) = self.model(torch.unsqueeze(x, 1), (state_h, state_c))

            condition = torch.tensor(1).to('cuda:0')
            loss = self.loss_fn(state_h[-1][-1], y, condition)

            # randomly sample negative examples
            # should really do contrastive loss over the batch
            for j in range(self.neg_samples):
                neg_idx = random.randint(0,len(X_train))
                if neg_idx == i: continue

                x_neg_query = X_train[j]
                x_neg_query_id = x_neg_query['query_id']
                x_neg = x_neg_query['encoding']
                y_neg = self.corpus[self.query_webpage_map[x_neg_query_id]]

                state_h, state_c = self.model.init_state()
                state_h = state_h.to('cuda:0')
                state_c = state_c.to('cuda:0')

                pred, (state_h, state_c) = self.model(torch.unsqueeze(x_neg, 1), (state_h, state_c))

                condition = torch.tensor(0).to('cuda:0')
                loss_neg = self.loss_fn(state_h[-1][0], y_neg, condition)

                loss += loss_neg
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            #if i % 1000 == 999:
            #    print('Inner loss: ', total_loss / (i*self.neg_samples))
        
        return total_loss / len(X_train * self.neg_samples)

    def dev(self, X_dev):
        total_loss = 0
        self.model.eval()
        for i,query in enumerate(X_dev):

            query_id = query['query_id']
            x = query['encoding']
            y = self.corpus[self.query_webpage_map[query_id]]

            state_h, state_c = self.model.init_state()
            state_h = state_h.to('cuda:0')
            state_c = state_c.to('cuda:0')

            pred, (state_h, state_c) = self.model(torch.unsqueeze(x, 1), (state_h, state_c))

            condition = torch.tensor(1).to('cuda:0')
            loss = self.loss_fn(state_h[-1][0], y, condition)

            total_loss += loss.item()
            
        return total_loss / len(X_dev)

    def test(self, X_test, k=10):
        run = []
        self.model.eval()

        # so that we can do cosine scores and recover the actual id of the webpage
        just_corpus_encodings = [torch.unsqueeze(self.corpus[x],dim=0) for x in self.corpus]
        just_corpus_encodings = torch.cat(just_corpus_encodings)

        webpage_id_map = {}
        for i,webpage_id in enumerate(self.corpus):
            webpage_id_map[i] = webpage_id


        for i,query in enumerate(X_test):

            query_id = query['query_id']
            x = query['encoding']

            state_h, state_c = self.model.init_state()
            state_h = state_h.to('cuda:0')
            state_c = state_c.to('cuda:0')
            pred, (state_h, state_c) = self.model(torch.unsqueeze(x, 1), (state_h, state_c))

            encoded_query = state_h[-1][0]

            cos_scores = util.cos_sim(encoded_query, just_corpus_encodings)[0]
            top_results = torch.topk(cos_scores, k=k)
            for j, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
                # 0 Q0 0 1 193.457108 Anserini
                run.append(' '.join([str(query_id), 'Q0', str(webpage_id_map[idx.item()]), str(j), str(score.item()), 'LSTM']))
        return run
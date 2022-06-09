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

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, 1, self.input_size),
                torch.zeros(self.num_layers, 1, self.input_size))


class LSTMTrainer():
    def __init__(self, model, loss_fn, optimizer, corpus, query_webpage_map, id_int_map, int_id_map, relevance_scores):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.corpus = corpus
        self.query_webpage_map = query_webpage_map
        self.id_int_map = id_int_map
        self.int_id_map = int_id_map
        self.relevance_scores = relevance_scores

        self.neg_samples = 1
        #TODO When need to make this hyperparameter, multiply by batch size
        self.margin = torch.tensor([10]).float().to('cuda:0')

    def train(self, X_train):
        total_loss = 0
        self.model.train()

        for i,query in enumerate(X_train):

            query_id = query['query_id']
            x = query['encoding']
            y = self.corpus[self.id_int_map[self.query_webpage_map[query_id]]]

            state_h, state_c = self.model.init_state(len(x))
            state_h = state_h.to('cuda:0')
            state_c = state_c.to('cuda:0')

            pred, (state_h, state_c) = self.model(torch.unsqueeze(x, 1), (state_h, state_c))
            
            condition = torch.tensor(1).to('cuda:0')
            loss = self.loss_fn(state_h[-1][0], y, condition)

            # scores = []
            # for query in run:
            #     ground_truth = relevance_scores[query]
            # if run[query][0] == ground_truth:
            #     scores.append(1)
            
            # return sum(scores) / len(query_ids)

            # for i in range(1):
            #     neg_idx = random.randint(0, len(X_train) - 1)
                
            #     if neg_idx == i: continue

            #     x_neg_query = X_train[neg_idx]
            #     x_neg = x_neg_query['encoding']
            #     x_neg_query_id = x_neg_query['query_id']
            #     y_neg = self.corpus[self.id_int_map[self.query_webpage_map[x_neg_query_id]]]

            #     state_h, state_c = self.model.init_state(len(x_neg))
            #     state_h = state_h.to('cuda:0')
            #     state_c = state_c.to('cuda:0')

            #     pred, (state_h, state_c) = self.model(torch.unsqueeze(x_neg, 1), (state_h, state_c))

            #     condition = torch.tensor(1).to('cuda:0')
            #     loss += self.loss_fn(state_h[-1][0], y_neg, condition)

            # condition = torch.tensor(0).to('cuda:0')
            # loss_neg = self.loss_fn(state_h[-1][0], y_neg, condition)
            # neg_idx = random.randint(0,len(X_train)-1)
            # x_neg_query = X_train[neg_idx]
            # x_neg_query_id = x_neg_query['query_id']
            
            # loss = self.loss_fn(pred[-1].squeeze(), y, y_neg, self.margin)
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            #if i % 1000 == 999:
            #    print('Inner loss: ', total_loss / (i*self.neg_samples))
        
        return total_loss / len(X_train)

    def val(self, X_val):
        total_loss = 0
        number_of_queries_processed = 0
        relevance_scores = []

        just_corpus_encodings = torch.unsqueeze(self.corpus[0], dim=0)

        for x in self.corpus[1:]:
            just_corpus_encodings = torch.cat((just_corpus_encodings, torch.unsqueeze(x, dim=0)), 0)

        self.model.eval()

        for i, query in enumerate(X_val):
            number_of_queries_processed += 1

            query_id = query['query_id']
            x = query['encoding']
            y = self.corpus[self.id_int_map[self.query_webpage_map[query_id]]]

            state_h, state_c = self.model.init_state(len(x))
            state_h = state_h.to('cuda:0')
            state_c = state_c.to('cuda:0')

            pred, (state_h, state_c) = self.model(torch.unsqueeze(x, 1), (state_h, state_c))

            condition = torch.tensor(1).to('cuda:0')
            loss = self.loss_fn(state_h[-1][0], y, condition)

            encoded_query = pred[-1].squeeze()

            scores = util.cos_sim(encoded_query, just_corpus_encodings)[0]
            top_results = torch.topk(scores, k=1)

            for j, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
                # 0 Q0 0 1 193.457108 Anserini
                ground_truth = self.query_webpage_map[query_id]
                if self.int_id_map[idx.item()] == ground_truth:
                    relevance_scores.append(1)

            if i % 1000 == 0:
                print(sum(relevance_scores) / number_of_queries_processed)

            # neg_idx = random.randint(0,len(X_val)-1)
            # x_neg_query = X_val[neg_idx]
            # x_neg_query_id = x_neg_query['query_id']
            # y_neg = self.corpus[self.id_int_map[self.query_webpage_map[x_neg_query_id]]]
            # condition = torch.tensor(1).to('cuda:0')
            # loss += self.loss_fn(state_h[-1][0], y_neg, condition)
            # loss = self.loss_fn(pred[-1].squeeze(), y, y_neg, self.margin)

            total_loss += loss.item()

        print("After the recent epoch: ")
        print(sum(relevance_scores) / number_of_queries_processed)
    
        return total_loss / len(X_val)

    def test(self, X_test, k=10):
        run = []
        self.model.eval()

        # so that we can do cosine scores and recover the actual id of the webpage

        just_corpus_encodings = torch.unsqueeze(self.corpus[0], dim=0)

        for x in self.corpus[1:]:
            just_corpus_encodings = torch.cat((just_corpus_encodings, torch.unsqueeze(x, dim=0)), 0)

        # just_corpus_encodings = [torch.unsqueeze(self.corpus[x],dim=0) for x in self.corpus]
        # just_corpus_encodings = torch.cat(just_corpus_encodings)

        webpage_id_map = {}
        for i, webpage_id in enumerate(self.corpus):
            webpage_id_map[i] = webpage_id

        for i, query in enumerate(X_test):

            query_id = query['query_id']
            x = query['encoding']

            state_h, state_c = self.model.init_state(len(x))
            state_h = state_h.to('cuda:0')
            state_c = state_c.to('cuda:0')
            pred, (state_h, state_c) = self.model(torch.unsqueeze(x, 1), (state_h, state_c))

            encoded_query = pred[-1].squeeze()

            scores = util.cos_sim(encoded_query, just_corpus_encodings)[0]
            top_results = torch.topk(scores, k=1)

            for j, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
                # 0 Q0 0 1 193.457108 Anserini
                # before it was webpage_id_map[i], but this is a tensor, we want the filename associated with that webpage encoding
                run.append(' '.join([str(query_id), 'Q0', str(self.int_id_map[idx.item()]), str(j), str(score.item()), 'LSTM']))

        return run
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

        self.linear1 = nn.Linear(hidden_size*2, hidden_size*2)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_size*2, hidden_size)


    def forward(self, x, prev_state, last_comment):
        pred, (state_h, state_c) = self.lstm(x, prev_state)
        merged = torch.cat((last_comment, state_h[-1][0]))
        out = self.linear1(merged)
        out = self.sigmoid(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, 1, self.input_size),
                torch.zeros(self.num_layers, 1, self.input_size))

class LSTMTrainer():
    def __init__(self, model, loss_fn, optimizer, corpus, query_webpage_map, id_int_map, int_id_map, relevance_scores, just_corpus_encodings, setting):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.corpus = corpus
        self.query_webpage_map = query_webpage_map
        self.id_int_map = id_int_map
        self.int_id_map = int_id_map
        self.relevance_scores = relevance_scores
        self.just_corpus_encodings = just_corpus_encodings
        self.setting = setting

        #TODO When need to make this hyperparameter, multiply by batch size
        # self.margin = torch.tensor([10]).float().to('cuda:0')

    def train(self, X_train, batch_size, num_of_negative_samples):
        total_loss = 0
        self.model.train()

        # experiment with different batch sizes
        for i, query_idx in enumerate(range(0, len(X_train), batch_size)):
            loss = 0
            for j in range(batch_size):
                if query_idx + j >= len(X_train):
                    break

                query = X_train[query_idx + j]
                query_id = query['query_id']
                x = query['encoding']

                # for proactive
                if self.setting == 'proactive':
                    if x.shape[0] != 1:
                        x = x[:-1, :]

                y = self.corpus[self.id_int_map[self.query_webpage_map[query_id]]]

                state_h, state_c = self.model.init_state(len(x))
                state_h = state_h.to('cuda:0')
                state_c = state_c.to('cuda:0')

                pred, (state_h, state_c) = self.model(torch.unsqueeze(x, 1), (state_h, state_c))
                
                condition = torch.tensor(1).to('cuda:0')
                loss += self.loss_fn(state_h[-1][0], y, condition)

                for i in range(num_of_negative_samples):
                    neg_idx = random.randint(0, len(X_train) - 1)
                    
                    if neg_idx == i: continue

                    x_neg_query = X_train[neg_idx]
                    x_neg_query_id = x_neg_query['query_id']
                    y_neg = self.corpus[self.id_int_map[self.query_webpage_map[x_neg_query_id]]]

                    condition = torch.tensor(-1).to('cuda:0')
                    loss += self.loss_fn(state_h[-1][0], y_neg, condition)
                    
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if i % 10000 == 999:
                print('Inner loss: ', total_loss / (i*self.neg_samples))
        
        return total_loss / len(X_train)

    def val(self, X_val, epoch, number_of_epochs, number_of_layers, batch_size, num_of_negative_samples, marg, learning_rate, warm_up_rate):
        total_loss = 0
        number_of_queries_processed = 0
        relevance_scores = []

        with open(f'{number_of_epochs}_{number_of_layers}_{batch_size}_{num_of_negative_samples}_{marg}_{learning_rate}_{warm_up_rate}.txt', 'a') as f:

            f.write(f"Epoch {epoch}:\n")
            self.model.eval()

            for i, query in enumerate(X_val):
                number_of_queries_processed += 1

                query_id = query['query_id']
                x = query['encoding']

                # for proactive
                if self.setting == 'proactive':
                    if x.shape[0] != 1:
                        x = x[:-1, :]    

                y = self.corpus[self.id_int_map[self.query_webpage_map[query_id]]]

                state_h, state_c = self.model.init_state(len(x))
                state_h = state_h.to('cuda:0')
                state_c = state_c.to('cuda:0')
                pred, (state_h, state_c) = self.model(torch.unsqueeze(x, 1), (state_h, state_c))
                condition = torch.tensor(1).to('cuda:0')
                loss = self.loss_fn(state_h[-1][0], y, condition)

                encoded_query = pred[-1].squeeze()
                scores = util.cos_sim(encoded_query, self.just_corpus_encodings)[0]
                top_results = torch.topk(scores, k=1)

                # calculate precision at 1
                for j, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
                    ground_truth = self.query_webpage_map[query_id]
                    if self.int_id_map[idx.item()] == ground_truth:
                        relevance_scores.append(1)

                if i % 1000 == 0 and i != 0:
                    f.write(f"{sum(relevance_scores) / number_of_queries_processed}\n")
                    print(sum(relevance_scores) / number_of_queries_processed)

                total_loss += loss.item()

            print("After the recent epoch: ")
            print(sum(relevance_scores) / number_of_queries_processed)
            p1_score_after_epoch = sum(relevance_scores) / number_of_queries_processed
            f.write(f"After the recent epoch: {p1_score_after_epoch}\n")
    
        return total_loss / len(X_val)

    def test(self, X_test, k=10):
        run = []
        self.model.eval()

        webpage_id_map = {}
        for i, webpage_id in enumerate(self.corpus):
            webpage_id_map[i] = webpage_id

        for i, query in enumerate(X_test):
            query_id = query['query_id']
            x = query['encoding']

            # for proactive
            if self.setting == 'proactive':
                if x.shape[0] != 1:
                    x = x[:-1, :]

            state_h, state_c = self.model.init_state(len(x))
            state_h = state_h.to('cuda:0')
            state_c = state_c.to('cuda:0')
            pred, (state_h, state_c) = self.model(torch.unsqueeze(x, 1), (state_h, state_c))

            encoded_query = pred[-1].squeeze()

            scores = util.cos_sim(encoded_query, self.just_corpus_encodings)[0]
            top_results = torch.topk(scores, k=10)

            for j, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
                run.append(' '.join([str(query_id), 'Q0', str(self.int_id_map[idx.item()]), str(j), str(score.item()), 'LSTM']))

        return run
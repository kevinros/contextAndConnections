from torch import nn
import torch
import math

import random

def train(dataset, model, lstm_init = False):
    total_loss = 0
    model.train()
    for i,(x,y) in enumerate(dataset):
        loss = 0
        if lstm_init:
            state_h, state_c = model.init_state(len(x))
            state_h = state_h.to('cuda:0')
            state_c = state_c.to('cuda:0')

            pred, (state_h, state_c) = model(torch.unsqueeze(x, 1), (state_h, state_c))

            condition = torch.tensor(1).to('cuda:0')
            loss = loss_fn(state_h[-1][0], y, condition)
        else:
            x = x.reshape(1, -1, 768)
            pred = model(x)
            condition = torch.tensor(1).to('cuda:0')
            loss = loss_fn(pred.flatten(), y, condition)


        # randomly sample negative examples
        # should really do contrastive loss over the batch
        for i in range(5):
            neg_idx = random.randint(0,len(X_train))
            if neg_idx == i: continue
            loss_neg = 0
            if lstm_init:
                (x_neg, y_neg) = dataset[i]

                state_h, state_c = model.init_state(len(x_neg))
                state_h = state_h.to('cuda:0')
                state_c = state_c.to('cuda:0')

                pred, (state_h, state_c) = model(torch.unsqueeze(x_neg, 1), (state_h, state_c))

                condition = torch.tensor(0).to('cuda:0')
                loss_neg = loss_fn(state_h[-1][0], y_neg, condition)
            else:
                (x_neg, y_neg) = dataset[i]
                x_neg = x_neg.reshape(1, -1, 768)

                pred = model(x_neg)

                condition = torch.tensor(0).to('cuda:0')
                loss_neg = loss_fn(pred.flatten(), y_neg, condition)

            loss += loss_neg
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataset)

def eval(X_test, Y_test, model, file, k=10, lstm_init = False):
    run = []
    model.eval()
    for i,x in enumerate(X_test):
        if lstm_init:
            state_h, state_c = model.init_state(len(x))
            state_h = state_h.to('cuda:0')
            state_c = state_c.to('cuda:0')
            pred, (state_h, state_c) = model(torch.unsqueeze(x, 1), (state_h, state_c))

            encoded_query = state_h[-1][0]
        else:
            x = x.reshape(1, -1, 768)
            pred = model(x)
            encoded_query = pred[-1][0]

        cos_scores = util.cos_sim(encoded_query, Y_test)[0]
        top_results = torch.topk(cos_scores, k=k)
        for j, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
            # 0 Q0 0 1 193.457108 Anserini
            run.append(' '.join([str(i), 'Q0', str(idx.item()), str(j), str(score.item()), 'LSTM']))
    with open(file, 'w') as f:
        for result in run:
            f.write(result + '\n')

import argparse
import json
import re
from sentence_transformers import SentenceTransformer
import pickle
import os

# For Semantic Baseline

# validation runs
# python3 encode_dataset.py --query_path ../data_2017-09/queries/queries_val.tsv --out ../data_2017-09/queries/queries_val_cos.pkl --model msmarco-distilbert-cos-v5 --model_name msmarco-distilbert-cos-v5
# python3 encode_dataset.py --query_path ../data_2017-09/queries_onlylast/queries_val.tsv --out ../data_2017-09/queries_onlylast/queries_val_cos.pkl --model msmarco-distilbert-cos-v5 --model_name msmarco-distilbert-cos-v5
# python3 encode_dataset.py --query_path ../data_2017-09/queries_removelast/queries_val.tsv --out ../data_2017-09/queries_removelast/queries_val_cos.pkl --model msmarco-distilbert-cos-v5 --model_name msmarco-distilbert-cos-v5

# test runs
# python3 encode_dataset.py --query_path ../data_2017-09/queries/queries_test.tsv --out ../data_2017-09/queries/queries_test_cos.pkl --model msmarco-distilbert-cos-v5 --model_name msmarco-distilbert-cos-v5
# python3 encode_dataset.py --query_path ../data_2017-09/queries_onlylast/queries_test.tsv --out ../data_2017-09/queries_onlylast/queries_test_cos.pkl --model msmarco-distilbert-cos-v5 --model_name msmarco-distilbert-cos-v5
# python3 encode_dataset.py --query_path ../data_2017-09/queries_removelast/queries_test.tsv --out ../data_2017-09/queries_removelast/queries_test_cos.pkl --model msmarco-distilbert-cos-v5 --model_name msmarco-distilbert-cos-v5

# webpage encodings
# python3 encode_dataset.py --webpages_path ../data_2017-09/webpages/ --out ../data_2017-09/encoded_webpages/ --model msmarco-distilbert-cos-v5 --model_name msmarco-distilbert-cos-v5



# the following are for semantic finetune, make sure to change data to correspond to model, and make sure to change out to match full,last,onlylast
# full
# python3 encode_dataset.py --query_path ../data_2017-09/queries/queries_val.tsv --out ../data_2017-09/queries/queries_val_2022-06-01_12-04-50.pkl --model_name 2022-06-01_12-04-50 --model ../out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries-2022-06-01_12-04-50
# python3 encode_dataset.py --query_path ../data_2017-09/queries/queries_test.tsv --out ../data_2017-09/queries/queries_test_2022-06-01_12-04-50.pkl --model_name 2022-06-01_12-04-50 --model ../out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries-2022-06-01_12-04-50
# python3 encode_dataset.py --webpages_path ../data_2017-09/webpages/ --out ../data_2017-09/encoded_webpages/ --model_name 2022-06-01_12-04-50 --model ../out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries-2022-06-01_12-04-50

# onlylast
# python3 encode_dataset.py --query_path ../data_2017-09/queries_onlylast/queries_val.tsv --out ../data_2017-09/queries_onlylast/queries_val_2022-06-02_21-24-49.pkl --model_name 2022-06-02_21-24-49 --model ../out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries_onlylast-2022-06-02_21-24-49
# python3 encode_dataset.py --query_path ../data_2017-09/queries_onlylast/queries_test.tsv --out ../data_2017-09/queries_onlylast/queries_test_2022-06-02_21-24-49.pkl --model_name 2022-06-02_21-24-49 --model ../out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries_onlylast-2022-06-02_21-24-49
# python3 encode_dataset.py --webpages_path ../data_2017-09/webpages/ --out ../data_2017-09/encoded_webpages/ --model_name 2022-06-02_21-24-49 --model ../out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries_onlylast-2022-06-02_21-24-49

# removelast
# python3 encode_dataset.py --query_path ../data_2017-09/queries_removelast/queries_val.tsv --out ../data_2017-09/queries_removelast/queries_val_2022-05-22_18-44-28.pkl --model_name 2022-05-22_18-44-28 --model ../out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries_removelast-2022-05-22_18-44-28
# python3 encode_dataset.py --query_path ../data_2017-09/queries_removelast/queries_test.tsv --out ../data_2017-09/queries_removelast/queries_test_2022-05-22_18-44-28.pkl --model_name 2022-05-22_18-44-28 --model ../out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries_removelast-2022-05-22_18-44-28
# python3 encode_dataset.py --webpages_path ../data_2017-09/webpages/ --out ../data_2017-09/encoded_webpages/ --model_name 2022-05-22_18-44-28 --model ../out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries_removelast-2022-05-22_18-44-28



# onlylast model but full, however per comment (for LSTM)
# python3 encode_dataset.py --query_path ../data_2017-09/queries/queries_train.tsv --out ../data_2017-09/queries/queries_train_2022-06-02_21-24-49_percomment.pkl --model_name 2022-06-02_21-24-49 --model ../out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries_onlylast-2022-06-02_21-24-49 --per_comment True
# python3 encode_dataset.py --query_path ../data_2017-09/queries/queries_val.tsv --out ../data_2017-09/queries/queries_val_2022-06-02_21-24-49_percomment.pkl --model_name 2022-06-02_21-24-49 --model ../out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries_onlylast-2022-06-02_21-24-49 --per_comment True
# python3 encode_dataset.py --query_path ../data_2017-09/queries/queries_test.tsv --out ../data_2017-09/queries/queries_test_2022-06-02_21-24-49_percomment.pkl --model_name 2022-06-02_21-24-49 --model ../out/semantic_finetune_runs/train_bi-encoder-mnrl-msmarco-distilbert-cos-v5-queries_onlylast-2022-06-02_21-24-49 --per_comment True



def process_query(src: str, per_comment=False) -> list:
    src = re.sub('\n', ' ', src)
    src = " ".join(src.split(' '))
    if per_comment:
        src = src.split('<C>')
    else:
        src = " <C> ".join(src.split('<C>')[::-1]) #reverses, so that only older comments are truncated
    return src

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help="path to query tsv file")
    parser.add_argument('--webpages_path', help="path webpage directory")
    parser.add_argument('--model', help="path to sentencetransformer model")
    parser.add_argument('--model_name', help="name to help distinguish encoding files")
    parser.add_argument('--out', help='path to save encodings')
    parser.add_argument('--per_comment', help='used for LSTM, if true, will return a list of individually embedded comments')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "2" #different from 1, which is where the model is training

    if args.model:
        sbert_model = SentenceTransformer(args.model)
        index_name = 'webpages_' + args.model_name
    else:
        sbert_model = SentenceTransformer('msmarco-distilbert-dot-v5')
        index_name = 'webpages_baseline'


    if args.query_path:
        if args.per_comment:
            per_comment = True
        else:
            per_comment = False
        encoded_queries = []
        with open(args.query_path, 'r') as f:
            for i,line in enumerate(f):
                split_line = line.split('\t')
                query_id = split_line[0]
                query = " ".join(split_line[1:])
                comments = process_query(query, per_comment=per_comment)
                encoded_queries.append({"query_id": query_id, 'encoding': sbert_model.encode(comments, convert_to_tensor=True)})

                if i % 1000 == 0:
                    print('Queries processed: ' + str(i))

        print('Total encoded queries :', len(encoded_queries))
        pickle.dump(encoded_queries, open(args.out, 'wb'))


    elif args.webpages_path:
        encoded_webpages = []
        encoded_webpage_ints = []
        int_id_map = {}
        for i,file in enumerate(os.listdir(args.webpages_path)):
            webpage = json.load(open(args.webpages_path + file, 'r'))
            encoded_webpage = sbert_model.encode(webpage['contents'], convert_to_tensor=True)

            encoded_webpages.append(encoded_webpage)
            encoded_webpage_ints.append(i)
            int_id_map[i] = webpage['id']

            if i % 1000 == 0:
                print('Webpages processed: ' + str(i))

        print('Total encoded webpages :', len(encoded_webpages))
        pickle.dump(encoded_webpages, open(args.out + index_name + '.pkl', 'wb'))
        pickle.dump(int_id_map, open(args.out + 'int_id_map_' + index_name + '.pkl', 'wb'))
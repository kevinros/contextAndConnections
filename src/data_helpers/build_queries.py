import argparse
import re
import pickle

# example run 
# python3 build_queries.py --data_path ../data_2017-09/ --out ../data_2017-09/queries/

def process_query(src: list, url: str, remove_last_comment=False) -> str:
    '''
    Processing query code
    '''
    if remove_last_comment:
        src = src[:-1]
    else:
        src[-1] = re.sub(re.escape(url), ' ', src[-1])

    processed_src = " <C> ".join(src)
    processed_src = re.sub('\[|\]', ' ', processed_src)
    processed_src = re.sub('\( \)', ' ', processed_src)
    if len(processed_src) > 1024:
        processed_src = processed_src[-1024:]
    processed_src = re.sub('\n', '', processed_src)
    return processed_src


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help="path to outputs of format_reddit_comments and scrape_urls")

    parser.add_argument('--out_path', help='dir path to save query sets and relevance scores')

    args = parser.parse_args()


    all_needed_comments = pickle.load(open(args.data_path + 'all_needed_comments.pkl', 'rb'))
    url_file_map = pickle.load(open(args.data_path + 'url_file_map.pkl', 'rb'))



    # build the queries
    query_idx = 0
    train_queries = []
    val_queries = []
    test_queries = []
    relevance_scores = []

    for url in url_file_map:
        if 'status' not in url_file_map[url] or url_file_map[url]['status'] == "failure": continue


        for chain in url_file_map[url]['chains']:
            comments = [all_needed_comments[id] for id in chain]
            processed_comments = process_query(comments, url)

            if url_file_map[url]['split'] == 'train':
                train_queries.append(str(query_idx) + '\t' + processed_comments)
            elif url_file_map[url]['split'] == 'val':
                val_queries.append(str(query_idx) + '\t' + processed_comments)
            elif url_file_map[url]['split'] == 'test':
                test_queries.append(str(query_idx) + '\t' + processed_comments)


            # build the relevance scores https://trec.nist.gov/data/qrels_eng/
            relevance_scores.append(str(query_idx) + ' 0 ' + url_file_map[url]['filename'] + ' 1')
            query_idx += 1

            if query_idx % 10 == 0:
                print('Queries processed: ' + str(query_idx))

    with open(args.out_path + 'queries_train.tsv', 'w') as f:
        for doc in train_queries:
            f.write(doc + '\n')
    with open(args.out_path + 'queries_val.tsv', 'w') as f:
        for doc in val_queries:
            f.write(doc + '\n')
    with open(args.out_path + 'queries_test.tsv', 'w') as f:
        for doc in test_queries:
            f.write(doc + '\n')

    with open(args.out_path + 'relevance_scores.txt', 'w') as f:
        for rs in relevance_scores:
            f.write(rs + '\n')


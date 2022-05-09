def load_rel_scores(path_to_rel_scores):
    # map from queryid to webpage filename
    relevance_scores = {}
    with open(path_to_rel_scores, 'r') as f:
        for line in f:
            split_line = line.split()
            relevance_scores[split_line[0]] = split_line[2]
    return relevance_scores

def load_run(path_to_run):
    # map from queryid to list of ordered webpages
    run = {}
    with open(path_to_run, 'r') as f:
        for line in f:
            line = line.split(' ')
            query_id = line[0]

            if query_id not in run:
                run[query_id] = []
            run[query_id].append(line[2])
    return run
    
# pyserini returns nothing for comments that are removed
# this is to cross-reference so we properly weight in scoring (e.g., make all their scores 0)
def load_query_ids(path_to_queries):
    queries = {}
    with open(path_to_queries, 'r') as f:
        for i,line in enumerate(f):
            split_line = line.split('\t')
            query_id = split_line[0]
            queries[query_id] = True
    return queries


def compute_mrr10(relevance_scores, run, query_ids):
    scores = []
    for query in run:
        ground_truth = relevance_scores[query]
        for i,returned_result in enumerate(run[query][:10]):
            if ground_truth == returned_result:
                scores.append(1 / (i+1))
    return sum(scores) / len(query_ids)

def compute_p1(relevance_scores, run, query_ids):
    scores = []
    for query in run:
        ground_truth = relevance_scores[query]
        if run[query][0] == ground_truth:
            scores.append(1)
    return sum(scores) / len(query_ids)



rs = {'0': 'a', '1': 'b', '2': 'c'}
ru = {'0': ['a','b','c'], '1': ['a','b','c'], '2': ['a','b','c']}
q_ids = {'0': True, '1': True, '2': True}
assert compute_mrr10(rs, ru, q_ids) == (1 + (1/2) + (1/3)) / 3
assert compute_p1(rs, ru, q_ids) == 1/3
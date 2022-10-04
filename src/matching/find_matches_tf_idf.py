import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_content(files, start, stop):
    corpus = []
    if (start >= len(files)):
        start = len(files) - 1
    if (stop >= len(files)):
        stop = len(files) - 1
    for i in range(start, stop):
        with open(files[i], "r") as f:
            entry = json.loads(f.read())
            corpus.append(entry["contents"])
            
    return corpus

def tf_idf_dots_(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    out = X @ X.T
    return out

def tf_idf_dots_comparitive(corpus1, corpus2):
    #only does upper triangle
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus1 + corpus2)
    out = np.array((X[0:len(corpus1)] @ X[len(corpus1):].T).todense())
    return out

def tf_idf_dots_block(block_start_1, block_start_2, block_size, threshold = .95):
    matches = {}
    corpus_1 = extract_content(files, block_start_1, block_size + block_start_1)
    corpus_2 = extract_content(files, block_start_2, block_size + block_start_2)
    mtx = tf_idf_dots_comparitive(corpus_1, corpus_2)
    coordinates = np.where(mtx > threshold)
    for k in range(len(coordinates[0])):
        x = coordinates[0][k]
        y = coordinates[1][k]
        if block_start_1+x == block_start_2+y:
            continue
        matches[(block_start_1+x,block_start_2+y)] = mtx[x][y]
        print((block_start_1+x,block_start_2+y))
    return matches

def tf_idf_dots_blocked(files, block_size, threshold = .95, fileloc="../matches.txt"):
    matches = {}
    for i in range(0, len(files), block_size):
        for j in range(0, len(files), block_size):
            block_match = tf_idf_dots_block(i, j, block_size)
            if block_match:
                with open(fileloc, "a+") as f:
                    for m in block_match:
                        matches[m] = block_match[m]
                        v,u = m
                        f.write("(" + str(v) +", " + str(u) + "): " + str(matches[(v,u)]) + "\n")
            if(j//block_size % 10 == 0):
                print((i,j))

def main_loop(di, block_size, threshold, fileloc):
    os.chdir(di)
    files = os.listdir()
    tf_idf_dots_blocked(files, block_size, threshold = threshold, fileloc = fileloc)
    
def tf_idf_dots_blocked_queries(corpus, block_size, threshold = .95, fileloc="matches_queries.txt"):
    matches = {}
    for i in range(0, len(corpus), block_size):
            for j in range(0, len(corpus), block_size):
                block_match = {}
                mtx = tf_idf_dots_comparitive(corpus[i:min(i+block_size, len(corpus))], corpus[j:min(j+block_size, len(corpus))])
                coordinates = np.where(mtx > threshold)
                for k in range(len(coordinates[0])):
                    x = coordinates[0][k]
                    y = coordinates[1][k]
                    if i+x == j+y:
                        continue
                    block_match[(i+x,j+y)] = mtx[x][y]
                with open(fileloc, "a+") as f:
                    for m in block_match:
                        matches[m] = block_match[m]
                        v,u = m
                        f.write("(" + str(v) +", " + str(u) + "): " + str(matches[(v,u)]) + "\n")

def main_loop_queries(di, block_size, threshold, fileloc):
    texts = list(pd.read_csv(di + "/queries_train.tsv", delimiter = "\t", names=["n", "text"])["text"]) + list(pd.read_csv(di + "/queries_val.tsv", delimiter = "\t", names=["n", "text"])["text"]) + list(pd.read_csv(di + "/queries_test.tsv", delimiter = "\t", names=["n", "text"])["text"])
    tf_idf_dots_blocked_queries(texts, block_size, threshold = threshold, fileloc = fileloc)

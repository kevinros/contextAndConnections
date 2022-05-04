import re
from urllib.request import urlopen
from bs4 import BeautifulSoup
import json
import requests
import pickle
import uuid
import signal
import argparse
import random

# example run 
# python3 scrape_urls.py --min_length 2 --data_path ../data_2017-09/

class TimeoutException(Exception):
        pass

def handler(signum, frame):
    raise TimeoutException

def scrape(url: str) -> str:
    """
    Given a URL, gets the plaintext of the webpage
    """
    try:
        html = urlopen(url).read()
    except Exception as e:
        return ""

    soup = BeautifulSoup(html, features="html.parser")

    # paragraphs = soup.find_all("p")
    # alphabet_checker = re.compile('[a-zA-z]')

    # for paragraph in paragraphs:
    #     paragraph_text = paragraph.get_text()
    #     if alphabet_checker.findall(paragraph_text):
    #         if 'wikipedia.org' in url:
    #             return paragraph_text

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines and short lines
    text = ""
    for chunk in chunks:
        if chunk and len(chunk) > 50:
            # inspired by https://bigscience.huggingface.co/blog/building-a-tb-scale-multilingual-dataset-for-language-modeling#:~:text=Filters%2C%20tools%2C%20and%20indicators%20of%20data%20quality
            only_text = re.sub('[^ \w\*]', '', chunk)
            if len(only_text) / len(chunk) > 0.8:
                text += chunk
    return text

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--min_length', help="minimum number of comments in chain to consider the URL")
    parser.add_argument('--data_path', help="path to output of format_reddit_comments.py")

    args = parser.parse_args()

    MIN_LENGTH = int(args.min_length)
    DATA_PATH = args.data_path

    TIMEOUT = 2
    random.seed(100)
    # first split into train, not train
    TRAIN_SIZE = 0.8
    # then split not train into val and test
    VAL_TEST_SPLIT = 0.5

    valid_urls = pickle.load(open(DATA_PATH + 'valid_urls.pkl', 'rb'))

    # First, map all unique URLs to a file hash
    url_file_map = {}
    for line in valid_urls:
        if len(valid_urls[line]['chain']) <= MIN_LENGTH:
            continue
        for url in valid_urls[line]['urls']:
            if url not in url_file_map:
                url_file_map[url] = {'filename': uuid.uuid4().hex, 'chains': [valid_urls[line]['chain']]}
            else:
                # takes care of case where one url has many comment chains
                # just make sure that we don't already have the chain (some duplicates in the comments)
                if valid_urls[line]['chain'] not in url_file_map[url]['chains']:
                    url_file_map[url]['chains'].append(valid_urls[line]['chain'])

    total_urls = len(url_file_map)
    print('Total webpages to fetch: ', total_urls)
    success = 0
    for i,url in enumerate(url_file_map):
        url_file_map[url]['status'] = 'failure'
       
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(TIMEOUT)
            text = scrape(url)
            signal.alarm(0) # disable alarm
        except:
            continue
        
        if len(text.strip()) > 100:
            if random.uniform(0,1) < TRAIN_SIZE:
                split="train"
            else:
                if random.uniform(0,1) < VAL_TEST_SPLIT:
                    split="val"
                else:
                    split="test"
            out = {'id': url_file_map[url]['filename'], 'contents': text}
            with open(DATA_PATH + 'webpages/' + url_file_map[url]['filename'] + '.json', 'w') as f:
                json.dump(out, f)
                url_file_map[url]['status'] = 'success'
                url_file_map[url]['split'] = split
                success += 1
        if i % 100 == 0: 
            print('Total URLs', i)
            print('Success', success)

    pickle.dump(url_file_map, open(DATA_PATH + 'url_file_map.pkl', 'wb'))


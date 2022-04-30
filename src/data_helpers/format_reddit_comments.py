import pickle
import json
import re
from urllib.parse import urlparse
import argparse


# example run
# python3 format_reddit_comments.py --out_path ../data_2017-09/ --raw_reddit_data ../data_2017-09/RC_2017-09

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', help="path to save extracted data")
    parser.add_argument('--raw_reddit_data', help="path to downloaded reddit comments")

    args = parser.parse_args()

    DATA_PATH = args.out_path
    RAW_PATH = args.raw_reddit_data


    # Domains to keep in valid_urls
    valid_domains = {'en.wikipedia.org': True, 'en.m.wikipedia.org': True, 'www.washingtonpost.com': True, 'www.theguardian.com': True, 'www.independent.co.uk': True,
            'www.theatlantic.com': True, 'www.bbc.com': True, 'www.nbcnews.com': True, 'www.usatoday.com': True, 'www.cnn.com': True, 'insider.foxnews.com': True,
            'www.npr.org': True, 'www.espn.com': True, 'www.politico.com': True, 'www.bbc.co.uk': True, 'www.telegraph.co.uk': True, 'www.businessinsider.com': True,
            'www.bloomberg.com': True, 'www.bbc.co.uk': True, 'www.forbes.com': True, 'abcnews.go.com': True, 'www.huffingtonpost.com': True, 'www.latimes.com': True,
            'www.pbs.org': True, 'www.thesun.co.uk': True, 'www.chicagotribune.com': True, 'www.dailymail.co.uk': True, 'www.cnbc.com': True, 'www.foxnews.com': True,
            'www.slate.com': True, 'www.wired.com': True, 'www.investopedia.com': True, 'www.theonion.com': True, 'www.vox.com': True, 'articles.chicagotribune.com': True}
    # URL endings to ignore
    ignore_type = {'pdf': True, 'jpg': True, 'png':True, 'gif':True}
    # Authors to ignore
    ignore_authors = {'HelperBot_': True}

    # Data structures to save
    all_urls = {}
    valid_urls = {}
    id_parent_mapping = {}

    # Go through all Reddit comments, and update all_urls, valid_urls 
    with open(RAW_PATH, 'r') as f:
        for i,line in enumerate(f):
            comment = json.loads(line)


            id_parent_mapping[comment['id']] = comment['parent_id'][3:]

            cleaned_urls = []
            cleaned_valid_urls = []
            for url in re.findall(r'(https?://\S+-\n)?(?(1)([\S]*)|(https?://\S+))', comment['body']):

                # heuristics for parsing url
                url = "".join(url)
                url = re.sub('\)', '', url)
                url = re.sub('\]', '', url)
                url = re.sub('\n', '', url)
                url = re.sub(',', '', url)
                url = re.sub(' ', '', url)
                if url[-1] == ".":
                    url = url[:-1]

                try:
                    domain = urlparse(url).netloc
                except:
                    continue

                # merge wikipedia and wikipedia.m domains
                if domain == "en.m.wikipedia.org":
                    url = re.sub('\.m', '', url)

                cleaned_urls.append(url)

                if url[-3:] in ignore_type: continue
                
                if domain not in valid_domains: continue
                
                cleaned_valid_urls.append(url)

            if cleaned_urls:
                all_urls[i] = {'urls': cleaned_urls}

            if cleaned_valid_urls:
                if comment['author'] in ignore_authors: continue
                valid_urls[i] = {'urls': cleaned_valid_urls, 'id': comment['id']}
                if len(valid_urls) % 10000 == 0: print('Valid URLs collected: ', len(valid_urls))

    print('All valid URLs collected')

    # Build the chains for each valid URL
    all_needed_comments = {}
    for line in valid_urls:
        id = valid_urls[line]['id']
        chain = []
        while id in id_parent_mapping:
            chain.append(id)
            all_needed_comments[id] = None
            id = id_parent_mapping[id]
        chain = chain[::-1]
        valid_urls[line]['chain'] = chain

    print('All id chains constructed')

    # Do a second pass-through to get all necessary comments
    with open(RAW_PATH, 'r') as f:
        for i,line in enumerate(f):
            comment = json.loads(line)
            comment_id = comment['id']
            if comment_id in all_needed_comments:
                all_needed_comments[comment_id] = comment['body']



    pickle.dump(all_needed_comments, open(DATA_PATH + 'all_needed_comments.pkl', 'wb'))
    pickle.dump(valid_urls, open(DATA_PATH + 'valid_urls.pkl', 'wb'))
    pickle.dump(all_urls, open(DATA_PATH + 'all_urls.pkl', 'wb'))
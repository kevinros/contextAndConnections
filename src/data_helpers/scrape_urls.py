import re
from urllib.request import urlopen
from bs4 import BeautifulSoup
import argparse
import json

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException

def scrape(url: str) -> str:
    '''
    Simple method to scrape text from URLs. Not very robust. Need to handle exceptions. YouTube links take very long
    '''
    try:
        html = urlopen(url).read()
    except Exception as e:
        return ""

    soup = BeautifulSoup(html, features="html.parser")

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


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--formatted', help='path to formatted json reddit comments')
        parser.add_argument('--prev', help="json of known URLs")
        parser.add_argument('--out', help='path to save scraped webpages')

        # example usage
        # python3 scrape_urls.py --formatted ../data/RC_2009-05_formatted.json --out ../data/RC_2009-05_webpages.json

        
        args = parser.parse_args()

        formatted = json.load(open(args.formatted, 'r'))

        if args.prev:
            known_urls = json.load(open(args.map, 'r'))
        else:
            known_urls = {}

        total_num = len(formatted)

        print('Total number of comment threads: ', total_num)

        for i,thread in enumerate(formatted):
            if i % 100 == 0: 
                print(i/total_num)
            url = thread['url']
            if url in known_urls:
                continue
            else:
                url_text = scrape(url)
                if url_text == "" or url_text == "\n":
                    continue
                known_urls[url] = url_text
            

        json.dump(known_urls, open(args.out, 'w'))
import requests
from bs4 import BeautifulSoup
import json

rgx = re.compile(r"(?:<a href=\")([^\"]*)(?:\"[^>]*>)([^<]*)(?:</a>)")

def get_loc(idn):
    comment = requests.get("https://hacker-news.firebaseio.com/v0/item/" + str(idn) + ".json").json()
    if comment is not None and "text" in comment.keys():
        soup = BeautifulSoup(comment["text"], "html.parser")
        comment["text_proc"] = soup.text
        links = [str(x) for x in soup.find_all("a")]
        comment["link_texts"] = [{"link": rgx.match(lk).group(1), "title_text": rgx.match(lk).group(2)} for lk in links]
    return comment

def get_top():
    return requests.get("https://hacker-news.firebaseio.com/v0/topstories.json").json()

def get_all_top():
    total_lookups = 0
    all_posts = []
    for i in get_top():
        post = recursive_lookup(i)
        all_posts.append(post)
        total_lookups += post["descendants"] + 1 if "descendants" in post.keys() else 1
    return (all_posts, total_lookups)

def write_to_file(data, fname):
    with open(fname, "w") as f:
        f.write(json.dumps(data))

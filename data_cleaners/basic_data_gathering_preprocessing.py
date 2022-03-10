import json
import regex as re

def load_json(file_loc: str) -> dict:
    #assume that json is converted into valid form as a list of objects
    with open("sample_data.json") as j:
        return json.load(j)

def count_links_in_body(comments: dict) -> int:
    linksh = re.compile("\[.*\]\(.*\)")
    num_posts_with_links = 0
    for comment in comments:
        if re.match(linksh, comment["body"]) is not None:
            num_posts_with_links += 1
    return num_posts_with_links

def indices_of_comments_with_links_in_body(comments: dict) -> int:
    linksh = re.compile("\[.*\]\(.*\)")
    indices = []
    for i, comment in enumerate(comments):
        if re.match(linksh, comment["body"]) is not None:
            indices.append(i)
    return indices

def count_comments_in_same_post_as_another(comments: dict) -> int:
    memo = {}
    count = 0
    for comment in comments:
        if comment["link_id"] in memo:
            count += 1
        else:
            memo[comment["link_id"]] = comment
    return count

def indicies_of_comments_by_post(comments: dict) -> int:
    memo = {}
    for i, comment in enumerate(comments):
        if comment["link_id"][3:] in memo: #remove "t3_" by starting at 4th char
            memo[comment["link_id"][3:]].append(i)
        else:
            memo[comment["link_id"][3:]] = [i]
    return memo

def count_chained_comments(comments: dict)-> int:
    count = 0
    memo = set()
    for i in comments:
        if i["id"] in memo:
            count += 1
    memo.add(i["parent_id"][3:])
    return count
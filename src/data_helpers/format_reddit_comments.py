import html2text
import json
import re
import argparse
import json
import tqdm

def collect_ancestors(comments: dict, comment_id: str) -> list:
    '''
    For a given list of comments and a comment in the list, reconstruct a path to the top-level comment
    Returns a list of comment IDs

    This method isn't very efficient but good enough for now
    '''
    ancestors = []
    while True:
        
        if comment_id[:2] == 't3': 
            # refers to a link (top-level comment)
            # means we've reached the top of the chain
            return ancestors[::-1]

        if comment_id[:2] == 't1':
            comment_id = comment_id[3:]

        try:
            # there is an error here sometimes where the comment id is not present in the list
            # probably fine for now, but may need to address in the future
            old_comment_id = comment_id
            comment_id = comments[comment_id]['parent_id']
            ancestors.append(old_comment_id)
        except:
            return ancestors[::-1]


def format_comments(comments_by_post: dict, domains: list) -> list:
    '''
    Cycle through all posts and comments, find any URL mentions, and save the mention location + the comment's ancestors

    Ignore URLs that are not in the domains pased in

    '''
    all_urls = []
    for post_id in comments_by_post:
        for comment_id in comments_by_post[post_id]:
    
            # check if post body contains URL, accounts for edge case when dash is at the end of the line
            current_comment_text = comments_by_post[post_id][comment_id]['body']
            urls = re.findall(r'(https?://\S+-\n)?(?(1)([\S]*)|(https?://\S+))', current_comment_text)
            
            if urls:
                ancestors = collect_ancestors(comments_by_post[post_id], comment_id)
                
                for url in urls:
                    url = "".join(list(url))



                    if len(ancestors) < 2:
                        continue


                    # heuristics for parsing errors
                    url = re.sub('\)', '', url)
                    url = re.sub('\]', '', url)
                    url = re.sub('\n', '', url)
                    url = re.sub('.', '', url)
                    url = re.sub(',', '', url)
                    url = re.sub(' ', '', url)

                    
                    # remove non-alphnumeric characters
                    url_letters = re.sub('[^0-9a-zA-Z]', '', url)
                                    
                    # ignore pdfs
                    if 'pdf' == url_letters[-3:] or 'jpg' in url_letters[-3:] or 'png' in url_letters[-3:] or 'gif' in url_letters[-3:]:
                        continue

                    for domain in domains:
                        if domain in url:
                           
                            context = []
                            for ancestor in ancestors:
                                context.append(comments_by_post[post_id][ancestor]['body'])
                                            
                            all_urls.append({"post_id": post_id, "comment_id": comment_id, "url": url, "ancestors": ancestors, "full_context": context})
                            break

    return all_urls


def load_and_format(path: str) -> dict:
    '''
    Load the reddit data
    '''
    comments_by_post = {}
    with open(path, 'r') as f:
        for line in f:
            d = json.loads(line)
            link_id = d['link_id']
            if link_id not in comments_by_post:
                comments_by_post[link_id] = {}
            d['body'] = html2text.html2text(d['body'])
            comments_by_post[link_id][d['id']] = d
            
    # only keep posts with more than 5 comments
    for key in list(comments_by_post.keys()):
        if len(comments_by_post[key]) < 5:
            del comments_by_post[key]

    return comments_by_post


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--raw', help='path to raw downloaded reddit comment data')
        parser.add_argument('--out', help='path to save processed comments')

        # example usage
        # python3 format_reddit_comments.py --raw ../data/RC_2009-05 --out ../data/RC_2009-05_formatted.json

        args = parser.parse_args()

        # this method takes the majority of the time
        comments_by_post = load_and_format(args.raw)

        domains = ['bbc.co.uk', 'cnn.com', 'wikipedia.org']

        formatted = format_comments(comments_by_post, domains)

        json.dump(formatted, open(args.out, 'w'))





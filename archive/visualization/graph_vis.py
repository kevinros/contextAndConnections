import networkx as nx
import json

def link_kid_posts_rec(parent, G):
    if "text_proc" in parent.keys():
        for label in parent["text_proc"]:
            if "kid_texts" in parent.keys():
                for child in parent["kid_texts"]:
                    if "text_proc" in child.keys():
                        for child_label in child["text_proc"]:
                            G.add_edge(label[0], child_label[0])
                link_kid_posts_rec(child, G)

def to_fdg_json(G, fname):
    nodes = []
    links = []
    for i in G.nodes:
        nodes.append({"id": i})
    for u,v in G.edges:
        links.append({"source" : u, "target" : v})
    write_to_file({"nodes" : nodes, "links": links}, fname)
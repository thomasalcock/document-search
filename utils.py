import json
import os
import math
import argparse
from typing import List

def count_n_docs_with_term(search_term: str, data: List[dict])-> int: 
    n_docs_with_term = 0
    for doc in data:
        if search_term in doc["cleaned_fact"]:
            n_docs_with_term += 1
    
    return n_docs_with_term
    
def count_occurences(search_term: str, word_collection: List[str]) -> List[int]:
   return sum([x == search_term for x in word_collection])

def avg_doc_len(data: List[dict]) -> float:
    lens = []
    for doc in data:
        lens.append(len(doc["cleaned_fact"]))
    return sum(lens) / len(data)     

def calculate_okapi_bm25(query: str, data: List[dict], 
                         k1: float = 1.2, b: float = 0.75):
    query = query.split()
    N = len(data)
    avg_len = avg_doc_len(data)

    for doc in data:
        doc_cleaned = doc["cleaned_fact"]
        doc_len = len(doc_cleaned)
        query_scores = []

        for word in query:
            nq = count_n_docs_with_term(word, data)
            idf = math.log(1 + (N - nq + 0.5) / (nq + 0.5))
            f = count_occurences(word, doc_cleaned)
            score = idf * (f * (k1+1) / (f + k1 * (1 - b + b * doc_len / avg_len)))
            query_scores.append(score)

        doc["bm25"] = sum(query_scores)


def calculate_idf(search_term: str, data: List[dict], n_docs: int) -> float:
    n_docs_with_term = count_n_docs_with_term(search_term, data)
    if n_docs_with_term == 0:
        exit(f"No search results found containing the word {search_term}")

    return math.log(n_docs / n_docs_with_term)


def calculate_tfs(search_term: str, data: List[dict], idf: float):
  for doc in data:
    split_str = doc["cleaned_fact"]
    n_occurences = count_n_docs_with_term(search_term, split_str)
    tf = n_occurences / len(split_str)
    doc["tf_idf"] = tf * idf


def load_data(data_dir):
    data = []
    for file in os.listdir(data_dir):
        file = os.path.join(data_dir, file) 
        with open(file, "r") as f:
            data.append(json.load(f))
    return data


def get_cli_args():
    parser = argparse.ArgumentParser(
    prog="Search Cat Facts",
    usage="python search_data --query <your query> --n <max_documents>",
    description="Searches the catfacts directory for documents matching the query"
    )

    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--method", type=str, required=False, 
                        default="bm25", choices=["bm25", "tf_idf", "min_lev"])
    parser.add_argument("--n", type=int, required=False, default=1)

    args = parser.parse_args()
    if len(args.query.split()) > 1 and args.method != "bm25":
        parser.error(f"Queries with more than one word only work with method bm25!")
    return args
    

def tail(string):
  return string[1:]

def head(string):
  return string[0]

def lev(a, b):
    if len(b) == 0:
        return len(a)
    elif len(a) == 0:
        return len(b)
    elif head(a) == head(b):
        return lev(tail(a), tail(b))
    else:
       return 1 + min(
          lev(tail(a), b),
          lev(a, tail(b)),
          lev(tail(a), tail(b))
       )
  

def calculate_min_lev(search_term: str,data: List[dict]):
    for doc in data:
        l_distances = [lev(search_term, word) for word in doc["cleaned_fact"]]
        min_lev = min(l_distances)
        doc["min_lev"] = min_lev


def print_results(n_results: int, data: List[dict], method: str):

    if method == "tf_idf":   
        data = sorted(data, key = lambda x: x[method], reverse=True)
    elif method == "min_lev":
        data = sorted(data, key = lambda x: x[method], reverse=False)
    elif method == "bm25":
        data = sorted(data, key = lambda x: x[method], reverse=True)
    else:
        raise NotImplementedError(f"Method {method} was not implemented!")

    i = 0
    while i < n_results:
        print("fact {}: {}, {} score {:.2f}".format(
            i+1, 
            data[i]["fact"],
            method, 
            data[i][method]
        ))
        i+=1



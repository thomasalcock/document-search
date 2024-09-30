from utils import *

#TODO: split words in preprocessing step
#TODO: separate pre-processing step from data download

if __name__ == "__main__":

  args = get_cli_args()
  
  search_term = args.query
  search_method = args.method
  n_results = args.n

  data = load_data(data_dir="catfacts")
  
  if search_method == "tf_idf":
    n_docs = len(data)
    idf = calculate_idf(search_term=args.query, data=data, n_docs=n_docs)
    calculate_tfs(search_term=args.query, data=data, idf=idf)
  
  elif search_method == "bm25":
    bm25_scores = calculate_okapi_bm25(args.query, data)
  
  elif search_method == "min_lev":
    calculate_min_lev(search_term=args.query, data=data)
  
  print_results(n_results=n_results, data=data, method=search_method)

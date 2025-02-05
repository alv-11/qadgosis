
import numpy as np
from importlib import reload
import urllib.request
from bs4 import BeautifulSoup
import os
import bz2
import gzip
import shutil
import rdflib
from rdflib import Graph
import networkx as nx
import gzip

 
output_dir = "../PKG_EXP/DBpedia/dbpedia_files"
output_new = "../PKG_EXP/DBpedia/dbpedia_subgraphs"



def dbpedia_download(output_dir):
    base_url = 'http://downloads.dbpedia.org/'

    data_files = [
        ("3.9/en/skos_categories_en.nt.bz2", "categories_5.nt.bz2"), #09/2013
        ("3.8/en/skos_categories_en.nt.bz2", "categories_4.nt.bz2"), #08/2012
        ("3.7/en/skos_categories_en.nt.bz2", "categories_3.nt.bz2"), # 08/2011
        ("3.6/en/skos_categories_en.nt.bz2", "categories_2.nt.bz2"), #01/20211
        ("3.5.1/en/skos_categories_en.nt.bz2", "categories_1.nt.bz2"),#04/2010
    ]
    os.makedirs(output_dir, exist_ok=True)

    #download and rename
    print('Starting file downloads...')
    for original_name, new_name in data_files:
        file_url = base_url + original_name
        output_file_path = os.path.join(output_dir, new_name)
        
        try:
            print(f"Downloading {file_url}...")
            urllib.request.urlretrieve(file_url, output_file_path)
        except Exception as e:
            print(f"Failed to download {file_url}. Error: {e}")




def change_bz2_to_gz(output_dir):
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".bz2"):
            bz2_file_path = os.path.join(output_dir, file_name)

            decompressed_file_path = bz2_file_path.replace(".bz2", "")
            gz_file_path = decompressed_file_path + ".gz"

            with bz2.BZ2File(bz2_file_path, "rb") as f_in:
                with open(decompressed_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            with open(decompressed_file_path, "rb") as f_in:
                with gzip.open(gz_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(bz2_file_path)
            os.remove(decompressed_file_path)



def extract_top_triples(input_dir, output_dir, max_triples=10000):
    """
    Extracts up to max_triples triples ensuring that only strongly connected components are considered.
    Saves them in `output_dir` with the same filename.
    """

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".nt.gz"): 
            input_file = os.path.join(input_dir, filename)  
            output_file = os.path.join(output_dir, filename)  
            
            g = rdflib.Graph()

            with gzip.open(input_file, 'rt') as f:
                g.parse(f, format='nt')

            nx_graph = nx.DiGraph()
            for s, p, o in g:
                nx_graph.add_edge(s, o)  

            # Find SCCs 
            sccs = list(nx.strongly_connected_components(nx_graph))

            sccs = [scc for scc in sccs if len(scc) > 1] 

            # Extract triples belonging to SCCs
            selected_triples = set()
            for scc in sccs:
                for s in scc:
                    for triple in g.triples((s, None, None)):
                        selected_triples.add(triple)
            
            selected_triples = list(selected_triples)[:max_triples]

            with gzip.open(output_file, 'wt') as out_f:
                for s, p, o in selected_triples:
                    out_f.write(f"{s} {p} {o} .\n")

            print(f"Processed: {filename} → Saved to {output_file} (strongly connected triples)")





if __name__ == "__main__":
    dbpedia_download(output_dir)
    change_bz2_to_gz(output_dir)

    #input_file = "/Users/au735195/REDI/BIKE_RENTAL/APEX_code/PKG_EXP/DBpedia/dbpedia_files/categories_1.nt.gz"  # Path to your original graph
    #target_entity = 'http--dbpedia.org-resource-CategoryFL-Class_Somalia_articles'
    extract_top_triples(output_dir, output_new, max_triples=100000)


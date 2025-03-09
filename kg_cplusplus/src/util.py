
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
import pickle
 
output_dir = "../PKG_EXP/DBpedia/dbpedia_files"
output_new = "../PKG_EXP/DBpedia/dbpedia_subgraphs"
output_int = "../PKG_EXP/DBpedia/dbpedia_subgraphs_int"



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


def load_mapping(mapping_file):
    if os.path.exists(mapping_file):
        with open(mapping_file, 'rb') as f:
            return pickle.load(f)
    return {}, {}, 0

def save_mapping(mapping_file, node_to_id, id_to_node, current_id):
    os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
    with open(mapping_file, 'wb') as f:
        pickle.dump((node_to_id, id_to_node, current_id), f)

def extract_top_triples_int(input_dir, output_dir, mapping_file, max_triples=100000):
    os.makedirs(output_dir, exist_ok=True)

    # Load mapping only once at the start
    node_to_id, id_to_node, current_id = load_mapping(mapping_file)
    print(f"Initial mapping size: {len(node_to_id)} nodes loaded.")

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".nt.gz"): 
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            g = rdflib.Graph()

            with gzip.open(input_file, 'rt') as f:
                g.parse(f, format='nt')

            nx_graph = nx.DiGraph()

            new_ids_assigned = 0  # Track new IDs for this timestep

            # Assign IDs while building the graph
            for s, p, o in g:
                for node in [s, p, o]:
                    if node not in node_to_id:
                        node_to_id[node] = current_id
                        id_to_node[current_id] = node
                        current_id += 1
                        new_ids_assigned += 1
                nx_graph.add_edge(node_to_id[s], node_to_id[o])

            # Find SCCs
            sccs = list(nx.strongly_connected_components(nx_graph))
            sccs = [scc for scc in sccs if len(scc) > 1]
            selected_triples = set()
            for scc in sccs:
                for s in scc:
                    original_node = id_to_node[s]  # Convert back to original URI if needed
                    for triple in g.triples((original_node, None, None)):
                        s_id = node_to_id[triple[0]]  # Subject ID as integer
                        p_id = node_to_id.get(triple[1], current_id)  # Predicate ID as integer
                        if triple[1] not in node_to_id:  # If not already assigned, assign a new ID
                            node_to_id[triple[1]] = current_id
                            current_id += 1
                        o_id = node_to_id[triple[2]]  # Object ID as integer
                        selected_triples.add((s_id, p_id, o_id))

        # Save the transformed triples to a file
            #output_triples_file = os.path.join(output_dir, "scc_int_triples.nt.gz")
            with gzip.open(output_file, 'wt') as out_f:
                for s_id, p_id, o_id in selected_triples:
                    out_f.write(f"{s_id} {p_id} {o_id} .\n")
            print(f"Transformed integer triples saved to {output_file}")


            # Debug output
            print(f"Processed: {filename} → Saved to {output_file}")
            print(f"New IDs assigned in this timestep: {new_ids_assigned}")
            print(f"Total nodes mapped so far: {len(node_to_id)}")
            if new_ids_assigned > 0:
                print(f"Sample of newly mapped nodes: {list(id_to_node.items())[-5:]}") 

            # Save updated mapping after each timestep
            save_mapping(mapping_file, node_to_id, id_to_node, current_id)

  


if __name__ == "__main__":
    #dbpedia_download(output_dir)
    #change_bz2_to_gz(output_dir)

    mapping_file = "mappings/id_mappings_new.pkl"
    extract_top_triples_int(output_dir, output_int, mapping_file,  max_triples=100000)




import numpy as np
from importlib import reload
import urllib.request
import os
import bz2
import gzip
import shutil
from rdflib import Graph
import networkx as nx
import gzip
import pickle
import rdflib
from rdflib.namespace import XSD  # Import XSD namespace
from collections import defaultdict


output_dir = "../PKG_EXP/DBpedia/dbpedia_files"
output_new = "../PKG_EXP/DBpedia/dbpedia_subgraphs"
output_int = "../PKG_EXP/DBpedia/dbpedia_subgraphs_intall"

# output_dir = "../PKG_EXP/DBpedia/dbpedia_files_mapping"
# output_new = "../PKG_EXP/DBpedia/dbpedia_subgraphs_mapping"
# output_int = "../PKG_EXP/DBpedia/dbpedia_subgraphs_article_int4"


#mappingbased_properties_en.nt
def dbpedia_download(output_dir):
    base_url = 'http://downloads.dbpedia.org/'

    data_files = [
        ("3.9/en/skos_categories_en.nt.bz2", "categories_5.nt.bz2"), #09/2013
        ("3.8/en/skos_categories_en.nt.bz2", "categories_4.nt.bz2"), #08/2012
        ("3.7/en/skos_categories_en.nt.bz2", "categories_3.nt.bz2"), # 08/2011
        ("3.6/en/skos_categories_en.nt.bz2", "categories_2.nt.bz2"), #01/20211
        ("3.5.1/en/skos_categories_en.nt.bz2", "categories_1.nt.bz2"),#04/2010
    ]


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



def extract_top_triples_int_(input_dir, output_dir, mapping_file, max_triples=100000):
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

                        if len(selected_triples) >= max_triples:
                            break

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

  
def extract_top_triples_int__(input_dir, output_dir, mapping_file):#, max_triples=None):
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

            # Process all triples in the graph
            selected_triples = set()
            for s, p, o in g:
                s_id = node_to_id[s]  # Subject ID as integer
                p_id = node_to_id.get(p, current_id)  # Predicate ID as integer
                if p not in node_to_id:  # If not already assigned, assign a new ID
                    node_to_id[p] = current_id
                    current_id += 1
                o_id = node_to_id[o]  # Object ID as integer
                selected_triples.add((s_id, p_id, o_id))

                # Stop if max_triples is reached
                #if len(selected_triples) >= max_triples:
                    #break

            # Save the transformed triples to a file
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



# Helper function to validate gYear
def is_valid_gyear(value):
    try:
        year = int(value)
        return 1 <= year <= 9999
    except ValueError:
        return False

# Helper function to validate date
def is_valid_date(value):
    try:
        from datetime import date
        parts = value.split("-")
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        date(year, month, day)  # This will raise an exception if the date is invalid
        return True
    except Exception:
        return False

# Preprocess RDF data to filter out invalid literals and malformed lines
def preprocess_rdf(input_file, output_file):
    with gzip.open(input_file, 'rt') as f_in, gzip.open(output_file, 'wt') as f_out:
        for line in f_in:
            # Skip empty lines, comment lines, or lines with only a period
            if not line.strip() or line.strip() == "." or line.strip().startswith("#"):
                print(f"Skipping malformed line: {line.strip()}")
                continue

            # Split the N-Triples line into subject, predicate, and object
            parts = line.strip().split(" ", 2)
            if len(parts) < 3:
                print(f"Skipping invalid line: {line.strip()}")
                continue

            subject, predicate, obj = parts[0], parts[1], parts[2]

            # Check if the object is a literal with a datatype
            if obj.startswith('"') and "^^" in obj:
                value, datatype = obj.rsplit("^^", 1)
                value = value.strip('"')
                datatype = datatype.strip("<>")

                # Handle gYear
                if datatype == str(XSD.gYear):
                    if not is_valid_gyear(value):
                        print(f"Replacing invalid gYear: {value} with '0001'")
                        value = "0001"  # Replace invalid gYear with a default value
                    obj = f'"{value}"^^<{datatype}>'

                # Handle date
                if datatype == str(XSD.date):
                    if not is_valid_date(value):
                        print(f"Skipping invalid date: {value}")
                        continue

            # Write the valid or modified triple to the output file
            f_out.write(f"{subject} {predicate} {obj} .\n")



# Extract triples and convert nodes to integers
def extract_top_triples_int(input_dir, output_dir, mapping_file, max_triples=100000):
    os.makedirs(output_dir, exist_ok=True)

    # Load mapping only once at the start
    node_to_id, id_to_node, current_id = load_mapping(mapping_file)
    print(f"Initial mapping size: {len(node_to_id)} nodes loaded.")

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".nt.gz"): 
            input_file = os.path.join(input_dir, filename)
            preprocessed_file = os.path.join(output_dir, f"preprocessed_{filename}")
            output_file = os.path.join(output_dir, filename)

            # Preprocess the RDF data to filter out invalid literals and malformed lines
            preprocess_rdf(input_file, preprocessed_file)

            g = rdflib.Graph()

  
            with gzip.open(input_file, 'rt') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line == '.':
                        #print(f"Skipping malformed line: {line}")
                        continue
                    try:
                        g.parse(data=line, format='nt')
                    except Exception as e:
                        continue
                        #print(f"Failed to parse line: {line}\nError: {e}")


            # with gzip.open(preprocessed_file, 'rt') as f:
            #     try:
            #         g.parse(f, format='nt')  # Parse preprocessed RDF data
            #     except Exception as e:
            #         print(f"Failed to parse {filename}: {e}")
            #         continue

            nx_graph = nx.DiGraph()

            new_ids_assigned = 0  # Track new IDs for this timestep

            # Assign IDs while building the graph
            for s, p, o in g:
                try:
                    for node in [s, p, o]:
                        if node not in node_to_id:
                            node_to_id[node] = current_id
                            id_to_node[current_id] = node
                            current_id += 1
                            new_ids_assigned += 1
                    nx_graph.add_edge(node_to_id[s], node_to_id[o])
                except Exception as e:
                    print(f"Skipping invalid triple ({s}, {p}, {o}): {e}")
                    continue

            # Process all triples in the graph
            selected_triples = set()
            for s, p, o in g:
                try:
                    s_id = node_to_id[s]  # Subject ID as integer
                    p_id = node_to_id.get(p, current_id)  # Predicate ID as integer
                    if p not in node_to_id:  # If not already assigned, assign a new ID
                        node_to_id[p] = current_id
                        current_id += 1
                    o_id = node_to_id[o]  # Object ID as integer
                    selected_triples.add((s_id, p_id, o_id))

                    # Stop if max_triples is reached
                    if len(selected_triples) >= max_triples:
                        break
                except Exception as e:
                    print(f"Skipping invalid triple ({s}, {p}, {o}): {e}")
                    continue

            # Save the transformed triples to a file
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


def extract_top_triples_int___(input_dir, output_dir, mapping_file, max_triples=100000):
    os.makedirs(output_dir, exist_ok=True)

    # Load mapping only once
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
            new_ids_assigned = 0

            for s, p, o in g:
                for node in [s, p, o]:
                    if node not in node_to_id:
                        node_to_id[node] = current_id
                        id_to_node[current_id] = node
                        current_id += 1
                        new_ids_assigned += 1
                nx_graph.add_edge(node_to_id[s], node_to_id[o])

            sccs = list(nx.strongly_connected_components(nx_graph))
            sccs = [scc for scc in sccs if len(scc) > 1]

            selected_triples = set()
            done = False

            for scc in sccs:
                for s in scc:
                    original_node = id_to_node[s]
                    for triple in g.triples((original_node, None, None)):
                        s_id = node_to_id[triple[0]]
                        p = triple[1]
                        o = triple[2]

                        if p not in node_to_id:
                            node_to_id[p] = current_id
                            id_to_node[current_id] = p
                            current_id += 1

                        p_id = node_to_id[p]
                        o_id = node_to_id[o]

                        selected_triples.add((s_id, p_id, o_id))
                        if len(selected_triples) >= max_triples:
                            done = True
                            break
                    if done:
                        break
                if done:
                    break

            with gzip.open(output_file, 'wt') as out_f:
                for s_id, p_id, o_id in selected_triples:
                    out_f.write(f"{s_id} {p_id} {o_id} .\n")

            print(f"Transformed integer triples saved to {output_file}")
            print(f"Processed: {filename}")
            print(f"New IDs assigned in this timestep: {new_ids_assigned}")
            print(f"Total nodes mapped so far: {len(node_to_id)}")
            if new_ids_assigned > 0:
                print(f"Sample of newly mapped nodes: {list(id_to_node.items())[-5:]}")

            save_mapping(mapping_file, node_to_id, id_to_node, current_id)

def extract_top_triples_int____(input_dir, output_dir, mapping_file, max_triples=100000):
    os.makedirs(output_dir, exist_ok=True)

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
            new_ids_assigned = 0

            for s, p, o in g:
                for node in [s, p, o]:
                    if node not in node_to_id:
                        node_to_id[node] = current_id
                        id_to_node[current_id] = node
                        current_id += 1
                        new_ids_assigned += 1
                nx_graph.add_edge(node_to_id[s], node_to_id[o])

            sccs = [scc for scc in nx.strongly_connected_components(nx_graph) if len(scc) > 1]

            selected_triples = set()
            done = False

            # Phase 1: Try to get triples from SCCs
            for scc in sccs:
                for s in scc:
                    original_node = id_to_node[s]
                    for triple in g.triples((original_node, None, None)):
                        s_id = node_to_id[triple[0]]
                        p = triple[1]
                        o = triple[2]

                        if p not in node_to_id:
                            node_to_id[p] = current_id
                            id_to_node[current_id] = p
                            current_id += 1

                        p_id = node_to_id[p]

                        if o not in node_to_id:
                            node_to_id[o] = current_id
                            id_to_node[current_id] = o
                            current_id += 1

                        o_id = node_to_id[o]

                        selected_triples.add((s_id, p_id, o_id))
                        if len(selected_triples) >= max_triples:
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
            # Phase 2: Pad if needed with high-degree bias
            if len(selected_triples) < max_triples:
                print("Padding with high-degree nodes to promote power-law structure.")
                degrees = defaultdict(int)

                for s, p, o in g:
                    degrees[s] += 1
                    degrees[o] += 1

                sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                used = set()
                for node, _ in sorted_nodes:
                    if node in used:
                        continue
                    for triple in g.triples((node, None, None)):
                        s, p, o = triple
                        s_id = node_to_id[s]

                        if p not in node_to_id:
                            node_to_id[p] = current_id
                            id_to_node[current_id] = p
                            current_id += 1
                        p_id = node_to_id[p]

                        if o not in node_to_id:
                            node_to_id[o] = current_id
                            id_to_node[current_id] = o
                            current_id += 1
                        o_id = node_to_id[o]

                        selected_triples.add((s_id, p_id, o_id))
                        used.add(s)
                        if len(selected_triples) >= max_triples:
                            break
                    if len(selected_triples) >= max_triples:
                        break

            # # Phase 2: Pad if needed
            # if len(selected_triples) < max_triples:
            #     for s, p, o in g:
            #         s_id = node_to_id[s]
            #         if p not in node_to_id:
            #             node_to_id[p] = current_id
            #             id_to_node[current_id] = p
            #             current_id += 1
            #         p_id = node_to_id[p]

            #         if o not in node_to_id:
            #             node_to_id[o] = current_id
            #             id_to_node[current_id] = o
            #             current_id += 1
            #         o_id = node_to_id[o]

            #         selected_triples.add((s_id, p_id, o_id))
            #         if len(selected_triples) >= max_triples:
            #             break

            # Save
            with gzip.open(output_file, 'wt') as out_f:
                for s_id, p_id, o_id in selected_triples:
                    out_f.write(f"{s_id} {p_id} {o_id} .\n")

            print(f"Transformed integer triples saved to {output_file}")
            print(f"Processed: {filename}")
            print(f"New IDs assigned in this timestep: {new_ids_assigned}")
            print(f"Total nodes mapped so far: {len(node_to_id)}")
            if new_ids_assigned > 0:
                print(f"Sample of newly mapped nodes: {list(id_to_node.items())[-5:]}")

            save_mapping(mapping_file, node_to_id, id_to_node, current_id)






# def is_powerlaw(graph, min_nodes=50):
#     if graph.number_of_nodes() < min_nodes:
#         return False
#     deg_seq = [d for _, d in graph.degree() if d > 0]
#     if len(deg_seq) < 2:
#         return False
#     try:
#         fit = Fit(deg_seq, verbose=False)
#         R, p = fit.distribution_compare('power_law', 'lognormal')
#         return p < 0.05 and R > 0
#     except Exception:
#         return False


# def extract_top_triples_int_new(input_dir, output_dir, mapping_file, max_triples=100000):
#     os.makedirs(output_dir, exist_ok=True)

#     node_to_id, id_to_node, current_id = load_mapping_(mapping_file)
#     print(f"Initial mapping size: {len(node_to_id)} nodes loaded.")

#     for filename in sorted(os.listdir(input_dir)):
#         if filename.endswith(".nt.gz"):
#             input_file = os.path.join(input_dir, filename)
#             output_file = os.path.join(output_dir, filename)

#             g = rdflib.Graph()
#             with gzip.open(input_file, 'rt') as f:
#                 g.parse(f, format='nt')

#             nx_graph = nx.DiGraph()
#             new_ids_assigned = 0

#             for s, p, o in g:
#                 for node in [s, p, o]:
#                     if node not in node_to_id:
#                         node_to_id[node] = current_id
#                         id_to_node[current_id] = node
#                         current_id += 1
#                         new_ids_assigned += 1
#                 nx_graph.add_edge(node_to_id[s], node_to_id[o])

#             sccs = [nx_graph.subgraph(c).copy() for c in nx.strongly_connected_components(nx_graph) if len(c) > 1]
#             sccs = sorted(sccs, key=lambda x: x.number_of_nodes(), reverse=True)

#             selected_triples = set()
#             done = False

#             for scc in sccs:
#                 if is_powerlaw(scc):
#                     print("Selected power-law SCC with", scc.number_of_nodes(), "nodes")
#                     for s in scc.nodes():
#                         original_node = id_to_node[s]
#                         for triple in g.triples((original_node, None, None)):
#                             s_id = node_to_id[triple[0]]
#                             p = triple[1]
#                             o = triple[2]

#                             if p not in node_to_id:
#                                 node_to_id[p] = current_id
#                                 id_to_node[current_id] = p
#                                 current_id += 1
#                             p_id = node_to_id[p]

#                             if o not in node_to_id:
#                                 node_to_id[o] = current_id
#                                 id_to_node[current_id] = o
#                                 current_id += 1
#                             o_id = node_to_id[o]

#                             selected_triples.add((s_id, p_id, o_id))
#                             if len(selected_triples) >= max_triples:
#                                 done = True
#                                 break
#                         if done:
#                             break
#                     if done:
#                         break

#             if len(selected_triples) < max_triples:
#                 print("Padding with high-degree nodes to promote power-law structure.")
#                 degrees = defaultdict(int)
#                 for s, p, o in g:
#                     degrees[s] += 1
#                     degrees[o] += 1

#                 sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
#                 used = set()
#                 for node, _ in sorted_nodes:
#                     if node in used:
#                         continue
#                     for triple in g.triples((node, None, None)):
#                         s, p, o = triple
#                         s_id = node_to_id[s]

#                         if p not in node_to_id:
#                             node_to_id[p] = current_id
#                             id_to_node[current_id] = p
#                             current_id += 1
#                         p_id = node_to_id[p]

#                         if o not in node_to_id:
#                             node_to_id[o] = current_id
#                             id_to_node[current_id] = o
#                             current_id += 1
#                         o_id = node_to_id[o]

#                         selected_triples.add((s_id, p_id, o_id))
#                         used.add(s)
#                         if len(selected_triples) >= max_triples:
#                             break
#                     if len(selected_triples) >= max_triples:
#                         break

#             with gzip.open(output_file, 'wt') as out_f:
#                 for s_id, p_id, o_id in selected_triples:
#                     out_f.write(f"{s_id} {p_id} {o_id} .\n")

#             print(f"Transformed integer triples saved to {output_file}")
#             print(f"Processed: {filename}")
#             print(f"New IDs assigned in this timestep: {new_ids_assigned}")
#             print(f"Total nodes mapped so far: {len(node_to_id)}")
#             if new_ids_assigned > 0:
#                 print(f"Sample of newly mapped nodes: {list(id_to_node.items())[-5:]}")

#             save_mapping_(mapping_file, node_to_id, id_to_node, current_id)

import os
import gzip
import rdflib
import networkx as nx
from collections import defaultdict
import powerlaw
import numpy as np

def is_powerlaw(graph):
    degrees = [deg for _, deg in graph.degree()]
    if len(degrees) < 10 or all(d <= 1 for d in degrees):
        return False
    try:
        fit = powerlaw.Fit(degrees, verbose=False)
        R, p = fit.distribution_compare('power_law', 'lognormal')
        return p > 0.1 and R > 0
    except Exception as e:
        print(f"Power-law fitting failed: {e}")
        return False

def extract_top_triples_int_new(input_dir, output_dir, mapping_file, max_triples=100000):
    os.makedirs(output_dir, exist_ok=True)

    node_to_id, id_to_node, current_id = load_mapping_(mapping_file)
    print(f"Initial mapping size: {len(node_to_id)} nodes loaded.")

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".nt.gz"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            g = rdflib.Graph()
            with gzip.open(input_file, 'rt') as f:
                g.parse(f, format='nt')

            nx_graph = nx.DiGraph()
            new_ids_assigned = 0

            for s, p, o in g:
                for node in [s, p, o]:
                    if node not in node_to_id:
                        node_to_id[node] = current_id
                        id_to_node[current_id] = node
                        current_id += 1
                        new_ids_assigned += 1
                nx_graph.add_edge(node_to_id[s], node_to_id[o])

            sccs = [nx_graph.subgraph(c).copy() for c in nx.strongly_connected_components(nx_graph) if len(c) > 1]
            sccs = sorted(sccs, key=lambda x: x.number_of_nodes(), reverse=True)

            selected_triples = set()
            done = False

            for scc in sccs:
                if is_powerlaw(scc):
                    print("Selected power-law SCC with", scc.number_of_nodes(), "nodes")
                    for s in scc.nodes():
                        original_node = id_to_node[s]
                        for triple in g.triples((original_node, None, None)):
                            s_id = node_to_id[triple[0]]
                            p = triple[1]
                            o = triple[2]

                            if p not in node_to_id:
                                node_to_id[p] = current_id
                                id_to_node[current_id] = p
                                current_id += 1
                            p_id = node_to_id[p]

                            if o not in node_to_id:
                                node_to_id[o] = current_id
                                id_to_node[current_id] = o
                                current_id += 1
                            o_id = node_to_id[o]

                            selected_triples.add((s_id, p_id, o_id))
                            if len(selected_triples) >= max_triples:
                                done = True
                                break
                        if done:
                            break
                    if done:
                        break

            if len(selected_triples) < max_triples:
                print("Padding with high-degree nodes to promote power-law structure.")
                degrees = defaultdict(int)
                for s, p, o in g:
                    degrees[s] += 1
                    degrees[o] += 1

                sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                used = set()
                for node, _ in sorted_nodes:
                    if node in used:
                        continue
                    for triple in g.triples((node, None, None)):
                        s, p, o = triple
                        if s not in node_to_id:
                            node_to_id[s] = current_id
                            id_to_node[current_id] = s
                            current_id += 1
                        s_id = node_to_id[s]

                        if p not in node_to_id:
                            node_to_id[p] = current_id
                            id_to_node[current_id] = p
                            current_id += 1
                        p_id = node_to_id[p]

                        if o not in node_to_id:
                            node_to_id[o] = current_id
                            id_to_node[current_id] = o
                            current_id += 1
                        o_id = node_to_id[o]

                        selected_triples.add((s_id, p_id, o_id))
                        used.add(s)
                        if len(selected_triples) >= max_triples:
                            break
                    if len(selected_triples) >= max_triples:
                        break

            with gzip.open(output_file, 'wt') as out_f:
                for s_id, p_id, o_id in selected_triples:
                    out_f.write(f"{s_id} {p_id} {o_id} .\n")

            print(f"Transformed integer triples saved to {output_file}")
            print(f"Processed: {filename}")
            print(f"New IDs assigned in this timestep: {new_ids_assigned}")
            print(f"Total nodes mapped so far: {len(node_to_id)}")
            if new_ids_assigned > 0:
                print(f"Sample of newly mapped nodes: {list(id_to_node.items())[-5:]}")

            save_mapping_(mapping_file, node_to_id, id_to_node, current_id)





import os
import gzip
import rdflib
import networkx as nx
from collections import defaultdict
from rdflib import URIRef
from powerlaw import Fit
from datetime import datetime
import matplotlib.pyplot as plt

def is_powerlaw_(graph):
    degrees = [d for n, d in graph.degree() if d > 0]
    if len(degrees) < 50:
        return False
    try:
        fit = Fit(degrees, verbose=False)
        return fit.power_law.alpha < 3.5
    except Exception:
        return False

def extract_top_triples_int_new_new(input_dir, output_dir, mapping_file, max_triples=100000):
    os.makedirs(output_dir, exist_ok=True)

    node_to_id, id_to_node, current_id = load_mapping_(mapping_file)
    print(f"Initial mapping size: {len(node_to_id)} nodes loaded.")

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".nt.gz"):
            continue

        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)

        g = rdflib.Graph()
        with gzip.open(input_file, 'rt') as f:
            g.parse(f, format='nt')

        nx_graph = nx.DiGraph()
        new_ids_assigned = 0

        degrees = defaultdict(int)
        for s, p, o in g:
            for node in [s, p, o]:
                if node not in node_to_id:
                    node_to_id[node] = current_id
                    id_to_node[current_id] = node
                    current_id += 1
                    new_ids_assigned += 1
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                nx_graph.add_edge(node_to_id[s], node_to_id[o])
                degrees[s] += 1
                degrees[o] += 1

        sccs = [nx_graph.subgraph(c).copy() for c in nx.strongly_connected_components(nx_graph) if len(c) > 1]
        sccs = sorted(sccs, key=lambda x: (x.number_of_nodes(), nx.degree_assortativity_coefficient(x)), reverse=True)

        selected_triples = set()
        done = False

        for scc in sccs:
            if is_powerlaw_(scc):
                print("Selected power-law SCC with", scc.number_of_nodes(), "nodes")
                for s in scc.nodes():
                    original_node = id_to_node[s]
                    for triple in g.triples((original_node, None, None)):
                        s_id = node_to_id[triple[0]]
                        p = triple[1]
                        o = triple[2]

                        for node in [p, o]:
                            if node not in node_to_id:
                                node_to_id[node] = current_id
                                id_to_node[current_id] = node
                                current_id += 1
                        p_id = node_to_id[p]
                        o_id = node_to_id[o]

                        selected_triples.add((s_id, p_id, o_id))
                        if len(selected_triples) >= max_triples:
                            done = True
                            break
                    if done:
                        break
                if done:
                    break

        if len(selected_triples) < max_triples:
            print("Padding with high-degree nodes to promote power-law structure.")
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            used = set()
            for node, _ in sorted_nodes:
                if node in used:
                    continue
                for triple in g.triples((node, None, None)):
                    s, p, o = triple
                    for n in [s, p, o]:
                        if n not in node_to_id:
                            node_to_id[n] = current_id
                            id_to_node[current_id] = n
                            current_id += 1
                    s_id = node_to_id[s]
                    p_id = node_to_id[p]
                    o_id = node_to_id[o]
                    selected_triples.add((s_id, p_id, o_id))
                    used.add(s)
                    if len(selected_triples) >= max_triples:
                        break
                if len(selected_triples) >= max_triples:
                    break

        with gzip.open(output_file, 'wt') as out_f:
            for s_id, p_id, o_id in selected_triples:
                out_f.write(f"{s_id} {p_id} {o_id} .\n")

        print(f"Transformed integer triples saved to {output_file}")
        print(f"Processed: {filename}")
        print(f"New IDs assigned in this timestep: {new_ids_assigned}")
        print(f"Total nodes mapped so far: {len(node_to_id)}")
        if new_ids_assigned > 0:
            print(f"Sample of newly mapped nodes: {list(id_to_node.items())[-5:]}")

        save_mapping_(mapping_file, node_to_id, id_to_node, current_id)
           

if __name__ == "__main__":
    #dbpedia_download(output_dir)
    #change_bz2_to_gz(output_dir)
    mapping_file = "mappings/id_intall_new.pkl"#"mappings/id_mappings_article.pkl"# "mappings/id_mappings_mappings.pkl"#
    extract_top_triples_int__(output_dir, output_int, mapping_file)
    #extract_top_triples_int_new(output_dir, output_int, mapping_file,  max_triples=10000)




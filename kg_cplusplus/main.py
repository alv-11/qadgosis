import os
import time
import random
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
import powerlaw
from src.base import DBPedia
from src.user import query_log_by_mids
from src.glimpse import SummaryMethod, GLIMPSE
from src.isummary import ISUMMARY
from src.metrics import total_query_log_metrics
from src.summary import _SummaryGraph__
import src.qadgosis.sankofa  as walks
from src.apex import APEX, DETAILED_LOGGING, APEX_
import pickle
import sys
from pathlib import Path
run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")




METHODS = { 
    'glimpse': SummaryMethod(GLIMPSE, 'GLIMPSE'),
    'glimpse-2': SummaryMethod(GLIMPSE, 'GLIMPSE-2', power=2),
    'apex': SummaryMethod(APEX, 'APEX'),
    'isummary': SummaryMethod(ISUMMARY, 'ISUMMARY'),

}


def flatten_list(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item


def sliding_window(window_size=2, training_set=None, summary=None, i=None, test_weights=None):
    data = []
    timesteps = sorted(training_set.keys())
    for j in range(len(timesteps) - window_size + 1):
        t1, t2 = timesteps[j:j + window_size]
        if t2 <= i and i != 0:
            t1_weight = test_weights[t1]
            t2_weight = test_weights[t2]
            data_t1 = np.random.choice(training_set[t1], size=int(len(training_set[t1]) * t1_weight), replace=False)
            data_t2 = np.random.choice(training_set[t2], size=int(len(training_set[t2]) * t2_weight), replace=False)
            combined_data = np.concatenate((data_t1, data_t2))
            total_F1, _, _ = total_query_log_metrics(summary, combined_data)
            data.append({"Timestep Window": f"{t1}-{t2}", "Total F1": total_F1})
    return pd.DataFrame(data)


def process_all_timesteps(training_set=None, summary=None, i=None, test_weights=None):
    if training_set is None or test_weights is None:
        print("Error: training_set and test_weights must be provided.")
        return None

    output_data = []
    data = []
    timesteps = sorted(training_set.keys())

    for t in timesteps:
        if t in test_weights:
            sampled_data = np.random.choice(training_set[t], size=int(len(training_set[t]) * test_weights[t]), replace=False)
            sampled_data = [q for q in sampled_data if isinstance(q, dict) and 'Parse' in q]
            if not sampled_data:
                continue
            data.extend(sampled_data)

    data = list(flatten_list(data))
    total_F1, _, _ = total_query_log_metrics(summary, data)
    output_data.append({"Timestep Window": f"{i}", "Total F1": total_F1})
    return pd.DataFrame(output_data)


##########################


def _normalize_weights(w):
    """Normalize a weight dict to sum to 1; handle zeros safely."""
    if not w:
        return {}
    tot = sum(w.values())
    if tot == 0:
        return {k: 0.0 for k in w}
    return {k: v / tot for k, v in w.items()}


def _safe_concat_df(list_of_dfs, rename_from="Timestep Window", rename_to="Timestep", value_name="Total F1"):
    """Concatenate small dataframes from process_all_timesteps; return empty df if none."""
    if not list_of_dfs:
        return pd.DataFrame(columns=[rename_to, value_name])
    df = pd.concat(list_of_dfs, ignore_index=True)
    if rename_from in df.columns:
        df = df.rename(columns={rename_from: rename_to})
    # Ensure columns exist
    if rename_to not in df.columns:
        df[rename_to] = pd.NA
    if value_name not in df.columns:
        df[value_name] = pd.NA
    # Drop duplicates by timestep
    if rename_to in df.columns:
        df = df.drop_duplicates(subset=rename_to)
    return df

# --------------------- process_summary_sankofa  ---------------------


def process_summary_sankofa(top_k_nodes_dict,  graphs, label, weights, global_pr_dict, test_data,
                            walks_runtime=None, max_triples=None,  n_priority=None):
    """
    Fully restored Sankofa processing:
    - Computes Future Test F1
    - Computes Weighted Future Test F1 (never NA)
    - Computes runtime (walks + summary)
    """

    triples_no = {}
    future_rows = []
    weighted_tests = []
    summary_runtime_rows = []

    print("Summarizing for SANKOFA")

    common_keys = top_k_nodes_dict.keys() & graphs.keys() & weights.keys() & test_data.keys()

    for key in sorted(common_keys):

        KG = graphs[key]
        top_k_nodes = {str(n): v for n, v in top_k_nodes_dict[key].items()}
        non_top_k_nodes = {str(n): v for n, v in global_pr_dict[key].items()}

        #eliminate top-k from global pr
        non_top_k_nodes =  {k:v for k,v in non_top_k_nodes.items() if k not in top_k_nodes}
        print(f"Creating summary for t={key} with {len(top_k_nodes)} top-k nodes and {len(non_top_k_nodes)} non-top-k nodes.")


        # -------- SUMMARY + RUNTIME --------
        t0 = time.time()

        S =  _SummaryGraph__(KG, top_k_nodes, non_top_k_nodes, max_triples)

        t1 = time.time()

        summary_runtime = t1 - t0
        summary_runtime_rows.append({"Timestep": key, "Runtime (summary)": summary_runtime})
        triples_no[key] = S.triple_count

        print(f"Summary graph for t={key} created in {summary_runtime:.2f}s")

        # PRESENT TEST F1
        log = test_data[key]
        total_F1, total_precision, total_recall = total_query_log_metrics(S, log)

        future_rows.append({"Timestep": key, "Future Test F1": total_F1})

        # HISTORICAL TESTS 
        test_weights = weights[key]
        total_w = sum(test_weights.values())
        test_weights_norm = {j: test_weights[j] / total_w for j in test_weights}

        df_weight = process_all_timesteps(
            training_set=test_data,
            summary=S,
            i=key,
            test_weights=test_weights_norm
        )
        weighted_tests.append(df_weight)



    # BUILD DATAFRAMES 
    df_future = pd.DataFrame(future_rows)
    df_summary_rt = pd.DataFrame(summary_runtime_rows)

    # Weighted Future Test F1
    if len(weighted_tests) > 0:
        df_weighted = pd.concat(weighted_tests, ignore_index=True)
        df_weighted = df_weighted.rename(columns={
            "Timestep Window": "Timestep",
            "Total F1": "Weighted Future Test F1"
        })
    else:
        df_weighted = pd.DataFrame(columns=["Timestep", "Weighted Future Test F1"])


    for df in (df_future, df_summary_rt, df_weighted):
        if not df.empty:
            df["Timestep"] = df["Timestep"].astype(int)

    if walks_runtime is None:
        walks_runtime = pd.DataFrame({"Timestep": sorted(common_keys),
                                      "Runtime (walks)": [0.0]*len(common_keys)})

    if "Runtime (walks)" not in walks_runtime.columns:
        walks_runtime["Runtime (walks)"] = 0.0

    # MERGE 
    merged = walks_runtime.merge(df_summary_rt, on="Timestep", how="right")
    merged = merged.merge(df_future, on="Timestep", how="left")
    merged = merged.merge(df_weighted, on="Timestep", how="left")

    # 
    merged["Weighted Future Test F1"] = merged["Weighted Future Test F1"].fillna(0.0)

    # TOTAL RUNTIME 
    merged["Runtime (seconds)"] = (
        merged["Runtime (summary)"].fillna(0.0) +
        merged["Runtime (walks)"].fillna(0.0)
    )

    # FINAL FORMAT 
    sankofa = merged[[
        "Timestep",
        "Runtime (seconds)",
        "Future Test F1",
        "Weighted Future Test F1"
    ]].copy()

    sankofa.to_csv("sankofa_results.csv", index=False)
    print(sankofa)

    return triples_no, sankofa



# --------------------- process_summary_glimpse  ---------------------
# 
def run_glimpse(summary_method, graphs, triples_no_mc, training_data, test_data, weights):
    glimpse_rows = []
    weighted_tests = []

    for t in sorted(graphs.keys() & triples_no_mc.keys() & weights.keys()):
        KG = graphs[t]
        K = triples_no_mc[t]
        norm_w = _normalize_weights(weights[t])

        # build merged logs (window N=2 as original)
        if t == 0:
            merged_logs = training_data[t]
        else:
            N = 2
            keys = sorted(training_data.keys())
            idx = keys.index(t)
            start = max(0, idx - N + 1)
            selected = keys[start: idx + 1]
            merged_logs = list(flatten_list([training_data[k] for k in selected]))

        t0 = time.time()
        S = summary_method(KG, K, merged_logs)
        run_time = time.time() - t0

        total_F1, _, _ = total_query_log_metrics(S, test_data[t])
        glimpse_rows.append({"Timestep": t, "Glimpse Present F1": total_F1, "Runtime (seconds)": run_time})

        df_w = process_all_timesteps(training_set=test_data, summary=S, i=t, test_weights=norm_w)
        weighted_tests.append(df_w)


    return glimpse_rows, weighted_tests


# --------------------- process_summary_APEX  ---------------------
def run_apex(graphs, triples_no_mc, training_data, test_data, weights):
    apex_rows = []
    weighted_tests = []

    H = q = e = r = csr = index_list = None

    for t in sorted(graphs.keys() & triples_no_mc.keys() & weights.keys()):
        KG = graphs[t]
        K = triples_no_mc[t]
        test_w = weights[t]  

        if t == 0:
            P, H, q, e, r, csr, index_list, total_F1, run_time = APEX(KG, K, training_data[t], test_data[t], t=t)
        else:
            try:
                P, H, q, e, r, csr, index_list, total_F1, run_time = APEX(
                    KG, K, training_data[t], test_data[t],
                    t=t, H=H, q=q, e=e, r=r,
                    csr_indirect_matrices=csr, index_list=index_list
                )
            except Exception:
                pass
                #fallback to full call if incremental fails
                P, H, q, e, r, csr, index_list, total_F1, run_time = APEX(KG, K, training_data[t], test_data[t], t=t)
        #total_F1, run_time, P = APEX_(KG, K,  training_data[t], test_data[t], t=t)

        apex_rows.append({"Timestep": t, "Apex Present F1": total_F1, "Runtime (seconds)": run_time})
        df_w = process_all_timesteps(training_set=test_data, summary=P, i=t, test_weights=_normalize_weights(test_w))
        weighted_tests.append(df_w)



    return apex_rows, weighted_tests


# --------------------- process_summary_ISUMMARY  ---------------------
def run_isummary(summary_method, graphs, triples_no_mc, training_data, test_data, weights):
    isummary_rows = []
    weighted_tests = []

    for t in sorted(graphs.keys() & triples_no_mc.keys() & weights.keys()):
        KG = graphs[t]
        K = triples_no_mc[t]
        norm_w = _normalize_weights(weights[t])

        if t == 0:
            t0 = time.time()
            S = ISUMMARY(KG, K, training_data[t])
            run_time = time.time() - t0
        else:
            N = 2
            keys = sorted(training_data.keys())
            idx = keys.index(t)
            start = max(0, idx - N + 1)
            selected = keys[start: idx + 1]
            merged_logs = list(flatten_list([training_data[k] for k in selected]))

            t0 = time.time()
            S = summary_method(KG, K, merged_logs)
            run_time = time.time() - t0

        total_F1, _, _ = total_query_log_metrics(S, test_data[t])
        isummary_rows.append({"Timestep": t, "Isummary Present F1": total_F1, "Runtime (seconds)": run_time})

        df_w = process_all_timesteps(training_set=test_data, summary=S, i=t, test_weights=norm_w)
        weighted_tests.append(df_w)


    return isummary_rows, weighted_tests


# Merge present and weighted results for each method
def _merge_present_and_weighted(present_rows, weighted_dfs, present_col_name, weighted_col_name, rename_weighted="Total F1"):
    present_df = pd.DataFrame(present_rows).drop_duplicates(subset="Timestep") if present_rows else pd.DataFrame(columns=["Timestep", "Runtime (seconds)", present_col_name])
    weighted_df = _safe_concat_df(weighted_dfs, rename_from="Timestep Window", rename_to="Timestep", value_name=rename_weighted)
    if not weighted_df.empty and rename_weighted in weighted_df.columns:
        weighted_df = weighted_df.rename(columns={rename_weighted: weighted_col_name})
   
    if not present_df.empty:
        present_df["Timestep"] = present_df["Timestep"].astype(int)
    if not weighted_df.empty and "Timestep" in weighted_df.columns:
        weighted_df["Timestep"] = weighted_df["Timestep"].astype(int)

    if not present_df.empty and not weighted_df.empty:
        merged = pd.merge(present_df, weighted_df, on="Timestep")
    else:
        merged = present_df if not present_df.empty else weighted_df
    return merged




def process_summary_baselines(graphs, triples_no_mc, training_data, test_data, methods, weights):
    """
    Orchestrates GLIMPSE / APEX / ISUMMARY runs and merges results.
    Returns: DataFrame (sota) with columns for each method.
    """

    summary_methods = [METHODS[name] for name in methods]


    raw_results = {
        "Glimpse": ([], []),
        "Apex": ([], []),
        "Isummary": ([], []),
    }

    # --- RUN METHODS ---
    for summary_method in summary_methods:
        name = summary_method.name_

        if name.startswith("GLIMPSE") and not name.startswith("APEX"):
            print("Summarizing for GLIMPSE")
            rows, weighted = run_glimpse(summary_method, graphs, triples_no_mc, training_data, test_data, weights)
            raw_results["Glimpse"][0].extend(rows)
            raw_results["Glimpse"][1].extend(weighted)

        elif name == "APEX":
            print("Summarizing for APEX")
            rows, weighted = run_apex(graphs, triples_no_mc, training_data, test_data, weights)
            raw_results["Apex"][0].extend(rows)
            raw_results["Apex"][1].extend(weighted)

        elif name == "ISUMMARY":
            print("Summarizing for ISUMMARY")
            rows, weighted = run_isummary(summary_method, graphs, triples_no_mc, training_data, test_data, weights)
            raw_results["Isummary"][0].extend(rows)
            raw_results["Isummary"][1].extend(weighted)

 
    merged_results = {}
    for method, (present, weighted) in raw_results.items():
        if not present:   
            continue

        merged = _merge_present_and_weighted(
            present,
            weighted,
            f"{method} Present F1",
            f"Weighted {method} F1"
        )

        if merged is None or merged.empty:
            continue


        merged.rename(columns={"Runtime (seconds)": f"{method} Runtime (seconds)"}, inplace=True)

        print(f"{method.upper()}\n", merged)
        merged.to_csv(f"{method.lower()}_results.csv", index=False)

        merged_results[method] = merged

    # --- FINAL MERGE ---
    dfs = list(merged_results.values())
    if not dfs:
        print("No results produced by any summary method.")
        return pd.DataFrame()

    sota = dfs[0]
    for df in dfs[1:]:
        sota = pd.merge(sota, df, on="Timestep", how="inner")

    return sota



def extract_snapshot_number(filename):
    base = os.path.basename(filename)
    num_part = base.replace("snapshot_", "").replace(".nt.gz", "")
    return int(num_part)

def load_snapshots(subgraph_dir):
    """Load all snapshots from the specified directory."""
    graphs = {}
    
    for i, file_name in enumerate(sorted(os.listdir(subgraph_dir))):
        path = os.path.join(subgraph_dir, file_name)
        KG = DBPedia(rdf_gz=path)
        KG.load()
        graphs[i] = KG
    return graphs

def analyze_graph_properties(graph, timestep):
    """Analyze and print properties of the graph."""
    G = nx.DiGraph()
    for s, p, o in graph.triples():
        G.add_edge(s, o, predicate=p, weight=0.01)
    
    

    degrees = [d for _, d in G.degree()]
    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
    R, p = fit.distribution_compare('power_law', 'lognormal')
    return G

def generate_query_logs(graph, timestep, num_queries=200):
    """Generate query logs for a given graph."""
    topic_mids = random.sample(list(graph.triples_.keys()), k=15)
    valid_topic_mids = [mid for mid in topic_mids if mid in graph.triples_]
    if not valid_topic_mids:
        print(f"No valid topic_mids found in KG for snapshot {timestep}")
        return None

    query_log = query_log_by_mids(
        graph,
        valid_topic_mids,
        num_queries,
        topic_dist=np.ones(len(valid_topic_mids)) / len(valid_topic_mids),
        shuffle=True,
        random_query_prob=0
    )
    return query_log

def save_or_load_query_log(query_log_file, graph, timestep):
    """Save or load query logs from a file."""
    if os.path.exists(query_log_file):
        print(f"Loading query_log from {query_log_file}")
        with open(query_log_file, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Generating query_log for snapshot {timestep}")
        query_log = generate_query_logs(graph, timestep)
        with open(query_log_file, "wb") as f:
            pickle.dump(query_log, f)
        return query_log

def process_snapshots(subgraph_dir):
    """Process all snapshots and generate graphs, query logs, and training/test data."""
    graphs, training_data, test_data, queries_ = {}, {}, {}, {}

    files = os.listdir(subgraph_dir)
    snapshot_files = sorted(
    [f for f in files if f.startswith("snapshot_") and f.endswith(".nt.gz")],
    key=extract_snapshot_number
        )

    print("Loading all subgraph snapshots...")

    for i, file_name in enumerate(snapshot_files):
        #file = "snapshot_1.nt.gz"
        path = os.path.join(subgraph_dir, file_name)
        KG = DBPedia(rdf_gz=path)
        KG.load()
        graphs[i] = KG

        query_log_file = f"query_log_snapshot_{i}.pkl"
        #query_log_file = "query_log_snapshot_100.pkl"
        query_log = save_or_load_query_log(query_log_file, KG, i)
        if not query_log:
            continue

        split_idx = int(0.8 * len(query_log))
        training_data[i] = query_log[:split_idx]
        test_data[i] = query_log[split_idx:]
        queries_[i] = [q['Parse']['TopicEntityMid'] for q in training_data[i]]

    return graphs, training_data, test_data, queries_

def main():
    dynagraph = {} 
    subgraph_dir ="/Users/au735195/Desktop/dbp"#"/Users/au735195/Downloads/KG_IMDB30"####"/Users/au735195/Desktop/wiki_1"############# ###"#../PKG_EXP/DBpedia/dbpedia_subgraphs_intall" ##" "/Users/au735195/Downloads/KG_snapshots"#
 
    # Load and process snapshots
    print("Loading all subgraph snapshots...")
    graphs, training_data, test_data, queries_ = process_snapshots(subgraph_dir)

    # Analyze graph properties
    for i, graph in graphs.items():
        G = analyze_graph_properties(graph, i)
        dynagraph[i] = G  

    # Convert graphs and queries for processing
    converted_dynagraph = []
    for i, (timestep, graph) in enumerate(dynagraph.items()):
        if i == 0:
            b = 0.001
            summary_size = int(len(graph.edges) * b) 
            k1 = (len(graph.nodes))**2 / len(graph.edges)
            K = int(k1 / (0.1/b)) 
            print(f"Summary size: {summary_size}, K: {K}, k1: {k1}, DENSITY: {len(graph.edges)/ (len(graph.nodes)*(len(graph.nodes)-1))}")

        # Convert the graph to the required format
        converted_graph = {
            int(node): {int(neighbor): float(graph[node][neighbor].get("weight", 0.0001)) 
                        for neighbor in graph.neighbors(node)}
            for node in graph.nodes
        }
        converted_dynagraph.append(converted_graph)


    converted_queries = {
        int(timestep): {int(i): int(query) for i, query in enumerate(query_list)}
        for timestep, query_list in queries_.items()
    }

    # Generate top-k summaries
    results = walks.generate_top_k_summary(
        num_timesteps=len(converted_dynagraph),
        num_walks=10,
        dynamic_graph=converted_dynagraph,
        queries=converted_queries,
        c=2,
        epsilon=0.25,
        k=K,
        decay_rate = 0.8
    )

    # Process summaries and evaluate
    top_k_cumsum_walks = results["top_k_cumsum_walks"]
    weights = results["all_weights"]
   
    runtime_per_timestep = results["runtime_per_timestep"]
    non_topk_global_pr =  results["global_pr"]#
    #print(non_topk_global_pr)

    runtime_df = pd.DataFrame(
        [{"Timestep": t, "Runtime (walks)": runtime} for t, runtime in runtime_per_timestep.items()]
    ).sort_values(by="Timestep")


    triples_no, sankofa = process_summary_sankofa(
        top_k_cumsum_walks,  graphs, label="Cumsum Walks", weights=weights, global_pr_dict=non_topk_global_pr,
        test_data=test_data, walks_runtime=runtime_df, max_triples=summary_size, n_priority=50
    )
    sankofa.to_csv("sankofa_results.csv", index=False)
    print(triples_no)


    # sota = process_summary_baselines(
    #     graphs, triples_no, training_data, test_data,
    #     methods=["apex"]#, "glimpse", "apex", "isummary"]
    #     , weights=weights
    # )
    # merged_df = pd.merge(sankofa, sota, on="Timestep")
    # merged_df.to_csv("all_results.csv", index=False)

if __name__ == "__main__":
    main()

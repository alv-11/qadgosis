
import time
import random
import numpy as np
import math
from src.base import KnowledgeGraph
import os
import time
import random
import numpy as np
import pandas as pd
import networkx as nx
import powerlaw
from src.base import DBPedia
from src.user import query_log_by_mids
from src.glimpse import SummaryMethod, GLIMPSE
from src.isummary import ISUMMARY
from src.metrics import total_query_log_metrics
from src.apex import APEX
import pickle




def weighted_topk(items, weights, k):
    """
    top-k using partial sort (largest weights).
    """

    keys = np.empty(len(weights), dtype=float)

    for i, w in enumerate(weights):
        if w <= 0:
            keys[i] = np.inf       
        else:
            u = random.random()
            keys[i] = -math.log(u) / w

    # get indices of k smallest keys
    idx = np.argpartition(keys, k)[:k]

    return [items[i] for i in idx]


class SummaryGraph(KnowledgeGraph):

    def __init__(self, KG, top_k_nodes, non_top_k_nodes, max_triples=None, batch_size=10):
        super().__init__()

        self.parent_ = KG
        self.top_k_nodes = top_k_nodes              
        self.top_k_set = set(top_k_nodes.keys())
        self.max_triples = max_triples
        self.batch_size = batch_size
        self.non_top_k_nodes = non_top_k_nodes

        self.triple_count = 0
        self.triples_set = set()
  

        # Reverse index restricted to top-k
        self.reverse_index = self._build_reverse_index_topk_only()

        # Internal triples 
        self._topk_neighbors = self._compute_topk_internal_triples()

        # Boundary candidates 
        cand, w = self._compute_boundary_candidates()

        remaining = max_triples - self.triple_count
        self.random_padding_order = weighted_topk(cand, w, remaining)
        self.random_padding_index = 0

        self._build_summary()


    # ---------------------------
    # Helpers
    # ---------------------------

    def parent(self):
        return self.parent_

    def _limit_reached(self):
        return self.max_triples is not None and self.triple_count >= self.max_triples

    def _try_add_triple(self, triple):
        if triple not in self.triples_set:
            self.add_triple(triple)
            self.triples_set.add(triple)
            self.triple_count += 1


    # ---------------------------
    # Reverse index
    # ---------------------------

    def _build_reverse_index_topk_only(self):
        reverse_idx = {}
        parent = self.parent_.triples_
        top_k = self.top_k_set

        for s, rel_dict in parent.items():
            for r, nbrs in rel_dict.items():
                for o in nbrs:
                    if o in top_k:
                        reverse_idx.setdefault(o, []).append((s, r))

        return reverse_idx


    # ---------------------------
    # INTERNAL TRIPLES (top-k × top-k)
    # ---------------------------

    def _compute_topk_internal_triples(self):
        parent = self.parent_.triples_
        rev = self.reverse_index
        scores = self.top_k_nodes
        top_k_nodes = self.top_k_set
        budget = self.max_triples

        internal = set() 

        # Collect internal triples
        for e1 in top_k_nodes:
            if e1 in parent:
                for r, nbrs in parent[e1].items():
                    for e2 in nbrs:
                        if e2 in top_k_nodes:
                            internal.add((e1, r, e2))

            if e1 in rev:
                for e2, r in rev[e1]:
                    if e2 in top_k_nodes:
                        internal.add((e2, r, e1))

        print(len(internal), budget, len(set(internal)))

   
        internal = list(internal)

        # # If small, keep all
        if len(internal) <= budget:
            return [(0.0, t) for t in internal]

        # Otherwise: score + partial select budget elements
        else:

            weights = [scores[t[0]] + scores[t[2]] for t in internal]
            warr = np.asarray(weights)
            idx = np.argpartition(-warr, budget)[:budget]

            selected = [(weights[i], internal[i]) for i in idx]
            return selected
    

    # ---------------------------
    # BOUNDARY TRIPLES
    # ---------------------------

    def _compute_boundary_candidates(self):
        """
        All (top-k node) boundary triples with weights.
        Deterministic partial ranking, no sampling.
        """
        parent = self.parent_.triples_
        rev = self.reverse_index
        scores = self.top_k_nodes
        other_scores = self.non_top_k_nodes
        top_k = self.top_k_set

        candidates = []
        weights = []

        for e1 in top_k:

            # Outgoing
            if e1 in parent:
                for r, nbrs in parent[e1].items():
                    for e2 in nbrs:
                        w = scores[e1] + other_scores.get(e2, 0.0)
                        candidates.append((e1, r, e2))
                        weights.append(w)

            # Incoming
            if e1 in rev:
                for e2, r in rev[e1]:
                    w = scores[e1] + other_scores.get(e2, 0.0)
                    candidates.append((e2, r, e1))
                    weights.append(w)


        return candidates, weights
    

    def _compute_fallback_candidates(self):

        parent = self.parent_.triples_
        seen = self.triples_set

        top_k = self.top_k_set
        scores_other = self.non_top_k_nodes

        candidates = []
        weights = []

        for s, rels in parent.items():

            if s in top_k:        # skip top-k nodes entirely
                continue

            for r, nbrs in rels.items():
                for o in nbrs:

                    if o in top_k:   # skip edges entering top-k region
                        continue

                    triple = (s, r, o)

                    if triple in seen:
                        continue

                    # weight based only on non-top-k scores
                    w = scores_other.get(s, 0.0) + scores_other.get(o, 0.0)

                    candidates.append(triple)
                    weights.append(w)

        return candidates, weights



    # ---------------------------
    # Summary builder
    # ---------------------------

    def _build_summary(self):

        # 1. Add internal triples
        for score, triple in self._topk_neighbors:
            if self._limit_reached():
                break
            self._try_add_triple(triple)


        # 2. Add boundary triples in batches
        while not self._limit_reached():

            added = self.add_related_random_triples_batch(self.batch_size)
            if not added:
                break


        #print("Adding fallback triples...")
        fallback_start= time.time()

        while not self._limit_reached():
        #print("Adding boundary triples...")
            fallback = self._fallback_add()
            if not fallback:
                break
        fallback_end = time.time()
        print("Fallback time:", fallback_end - fallback_start)

    # ---------------------------
    # Batched addition
    # ---------------------------

    def add_related_random_triples_batch(self, batch_size=100):
        order = self.random_padding_order
        idx = self.random_padding_index
        n = len(order)

        if idx >= n:
            return False

        remaining = self.max_triples -  self.triple_count
        k = min(batch_size, remaining)

        end = min(idx + k, n)
        batch = order[idx:end]

        added = False
        for triple in batch:
            if self._limit_reached():
                break
            if triple not in self.triples_set:
                self._try_add_triple(triple)
                added = True

        self.random_padding_index = end
        return added


    def _fallback_add(self):
        remaining = self.max_triples - self.triple_count
        if remaining <= 0:
            return

        cand, w = self._compute_fallback_candidates()
        if not cand:
            return

        arr = np.asarray(w, dtype=float)
        k = min(remaining, len(arr))

        idx = np.argpartition(-arr, k)[:k]

        for i in idx:
            if self._limit_reached():
                break
            self._try_add_triple(cand[i])



METHODS = { 
    'glimpse': SummaryMethod(GLIMPSE, 'GLIMPSE'),
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
                            walks_runtime=None, max_triples=None):
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

        S =  SummaryGraph(KG, top_k_nodes, non_top_k_nodes, max_triples)

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
        path = os.path.join(subgraph_dir, file_name)
        KG = DBPedia(rdf_gz=path)
        KG.load()
        graphs[i] = KG

        query_log_file = f"query_log_snapshot_{i}.pkl"
        query_log = save_or_load_query_log(query_log_file, KG, i)
        if not query_log:
            continue

        split_idx = int(0.8 * len(query_log))
        training_data[i] = query_log[:split_idx]
        test_data[i] = query_log[split_idx:]
        queries_[i] = [q['Parse']['TopicEntityMid'] for q in training_data[i]]

    return graphs, training_data, test_data, queries_

import time
import random
import numpy as np
import math
import hashlib
import statistics
from src.base import KnowledgeGraph
import os
import time
import pandas as pd
import networkx as nx
from src.base import DBPedia
from src.user import query_log_by_mids
from src.glimpse import SummaryMethod, GLIMPSE
from src.hhop_random import HHOP_RANDOM
from src.isummary import ISUMMARY
from src.metrics import total_query_log_metrics, average_query_log_metrics
from src.apex import APEX
from src.pagerank import PPR_FREQUENCY
import json
import pickle
import re
import shutil
from concurrent.futures import ThreadPoolExecutor

from src.progress import format_seconds, progress, progress_write


ENABLE_SUMMARY_NODE_INVOLVEMENT_STATS = True



def _extract_seed_nodes_from_query(query):
    """Return Sankofa seed nodes from answer entities, fallback to topic entity."""
    parse = query.get("Parse", {}) if isinstance(query, dict) else {}
    answers = parse.get("Answers", [])

    seed_nodes = []
    for ans in answers:
        if not isinstance(ans, dict):
            continue
        if ans.get("AnswerType") != "Entity":
            continue
        answer_arg = ans.get("AnswerArgument")
        if answer_arg is not None:
            seed_nodes.append(answer_arg)

    if seed_nodes:
        return seed_nodes

    topic_entity = parse.get("TopicEntityMid")
    return [topic_entity] if topic_entity is not None else []


def _build_seed_node_list(training_queries):
    seed_nodes = []
    for query in training_queries:
        seed_nodes.extend(_extract_seed_nodes_from_query(query))
    return seed_nodes


def weighted_sampling(items, weights, k):
    """
    ES (Efraimidis–Spirakis) weighted sampling without replacement.
    """

    n = len(weights)
    if k <= 0 or n == 0:
        return []

    k = min(k, n)
    if k == n:
        return list(items)

    keys = np.empty(n, dtype=float)

    for i, w in enumerate(weights):
        if w <= 0:
            keys[i] = np.inf
        else:
            u = random.random()
            keys[i] = -math.log(u) / w

    idx = np.argpartition(keys, k - 1)[:k]
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
        self.internal_candidate_count = 0
        self.boundary_candidate_count = 0
        self.boundary_sampled_count = 0
        self.random_candidate_count = 0
        self._node_involvement_stats_cache = None
        self.stage_counts = {
            "internal": 0,
            "boundary": 0,
            "random": 0,
        }
  

        # Reverse index restricted to top-k
        self.reverse_index = self._build_reverse_index_topk_only()
        self.reverse_index_link_count = sum(len(pairs) for pairs in self.reverse_index.values())

        # Internal triples 
        self._topk_neighbors = self._compute_topk_internal_triples()

        # Boundary candidates 
        cand, w = self._compute_boundary_candidates()

        remaining = max_triples - self.triple_count
        self.random_padding_order = weighted_sampling(cand, w, remaining)
        self.boundary_sampled_count = len(self.random_padding_order)
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
            self._node_involvement_stats_cache = None

    def _node_triple_involvement_stats(self):
        if self._node_involvement_stats_cache is not None:
            return self._node_involvement_stats_cache

        counts = {}
        for subject, _, obj in self.triples_set:
            counts[subject] = counts.get(subject, 0) + 1
            if obj != subject:
                counts[obj] = counts.get(obj, 0) + 1

        if not counts:
            stats = {"mean": 0.0, "median": 0.0, "mode": 0}
        else:
            values = list(counts.values())
            modes = statistics.multimode(values)
            stats = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "mode": min(modes) if modes else 0,
            }

        self._node_involvement_stats_cache = stats
        return stats

    def _stage_percentage(self, stage_name):
        if self.triple_count == 0:
            return 0.0
        return 100.0 * self.stage_counts[stage_name] / self.triple_count

    def summary_overview(self):
        overview = (
            f"triples={self.triple_count} entities={self.number_of_entities()} "
            f"topk_nodes={len(self.top_k_set)} reverse_index_targets={len(self.reverse_index)} "
            f"reverse_index_links={self.reverse_index_link_count} "
            f"internal={self.stage_counts['internal']}/{self.internal_candidate_count} "
            f"({self._stage_percentage('internal'):.1f}%) "
            f"boundary={self.stage_counts['boundary']}/{self.boundary_sampled_count} "
            f"sampled_from={self.boundary_candidate_count} "
            f"({self._stage_percentage('boundary'):.1f}%) "
            f"random={self.stage_counts['random']}/{self.random_candidate_count} "
            f"({self._stage_percentage('random'):.1f}%)"
        )

        if ENABLE_SUMMARY_NODE_INVOLVEMENT_STATS:
            involvement = self._node_triple_involvement_stats()
            overview += (
                f" node_triples(mean={involvement['mean']:.2f}, "
                f"median={involvement['median']:.2f}, mode={involvement['mode']})"
            )

        return overview

    def composition_stats(self, timestep=None):
        stats = {
            "Summary Triple Count": self.triple_count,
            "Summary Entity Count": self.number_of_entities(),
            "TopK Node Count": len(self.top_k_set),
            "Summary Budget": self.max_triples,
            "Reverse Index Target Count": len(self.reverse_index),
            "Reverse Index Link Count": self.reverse_index_link_count,
            "Internal Triple Count": self.stage_counts["internal"],
            "Internal Candidate Triple Count": self.internal_candidate_count,
            "Internal Triple Percentage": self._stage_percentage("internal"),
            "Boundary Triple Count": self.stage_counts["boundary"],
            "Boundary Candidate Triple Count": self.boundary_candidate_count,
            "Boundary Sampled Triple Count": self.boundary_sampled_count,
            "Boundary Triple Percentage": self._stage_percentage("boundary"),
            "Random Triple Count": self.stage_counts["random"],
            "Random Candidate Triple Count": self.random_candidate_count,
            "Random Triple Percentage": self._stage_percentage("random"),
        }

        if ENABLE_SUMMARY_NODE_INVOLVEMENT_STATS:
            involvement = self._node_triple_involvement_stats()
            stats.update({
                "Node Triple Involvement Mean": involvement["mean"],
                "Node Triple Involvement Median": involvement["median"],
                "Node Triple Involvement Mode": involvement["mode"],
            })

        if timestep is not None:
            stats = {"Timestep": int(timestep), **stats}
        return stats


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


   
        internal = list(internal)
        self.internal_candidate_count = len(internal)

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

        self.boundary_candidate_count = len(candidates)

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
        initial_triple_count = self.triple_count

        # 1. Add internal triples
        for score, triple in self._topk_neighbors:
            if self._limit_reached():
                break
            self._try_add_triple(triple)
        self.stage_counts["internal"] += self.triple_count - initial_triple_count


        # 2. Add boundary triples in batches
        while not self._limit_reached():
            added = self.add_related_random_triples_batch(self.batch_size)
            if not added:
                break
            self.stage_counts["boundary"] += added


        #print("Adding fallback triples...")

        while not self._limit_reached():
            fallback = self._fallback_add()
            if not fallback:
                break
            self.stage_counts["random"] += fallback


    # ---------------------------
    # Batched addition
    # ---------------------------

    def add_related_random_triples_batch(self, batch_size=100):
        order = self.random_padding_order
        idx = self.random_padding_index
        n = len(order)

        if idx >= n:
            return 0

        remaining = self.max_triples -  self.triple_count
        k = min(batch_size, remaining)

        end = min(idx + k, n)
        batch = order[idx:end]

        added = 0
        for triple in batch:
            if self._limit_reached():
                break
            if triple not in self.triples_set:
                self._try_add_triple(triple)
                added += 1

        self.random_padding_index = end
        return added



    def _fallback_add(self):
        remaining = self.max_triples - self.triple_count
        if remaining <= 0:
            return 0

        cand, w = self._compute_fallback_candidates()
        if not cand:
            return 0

        self.random_candidate_count = max(self.random_candidate_count, len(cand))

        selected = weighted_sampling(cand, w, remaining)
        added = 0

        for triple in selected:
            if self._limit_reached():
                break
            if triple not in self.triples_set:
                self._try_add_triple(triple)
                added += 1

        return added


def _log_sankofa_summary(timestep, summary):
    progress_write(f"Sankofa summary t={timestep}: {summary.summary_overview()}")




METHODS = { 
    'glimpse': SummaryMethod(GLIMPSE, 'GLIMPSE'),
    'apex': SummaryMethod(APEX, 'APEX'),
    'isummary': SummaryMethod(ISUMMARY, 'ISUMMARY'),
    'hhop_random': SummaryMethod(HHOP_RANDOM, 'HHOP_RANDOM', h=2, neighborhood_mode='both'),
    'hhop_random_outgoing': SummaryMethod(HHOP_RANDOM, 'HHOP_RANDOM_OUTGOING', h=2, neighborhood_mode='outgoing_only'),
    'ppr_frequency': SummaryMethod(PPR_FREQUENCY, 'PPR_FREQUENCY', n=5),

}


_METHOD_LABELS = {
    'glimpse': 'Glimpse',
    'apex': 'Apex',
    'isummary': 'Isummary',
    'hhop_random': 'H-Hop Random',
    'hhop_random_outgoing': 'H-Hop Random Outgoing',
    'ppr_frequency': 'PPR Frequency',
}


def _validated_method_names(methods):
    if methods is None:
        return []

    method_names = list(dict.fromkeys(methods))
    unknown = [name for name in method_names if name not in METHODS]
    if unknown:
        raise ValueError(f"Unknown baseline methods: {', '.join(unknown)}")

    return method_names


def _baseline_label(method_name):
    return _METHOD_LABELS.get(method_name, method_name.replace('_', ' ').title())


def _present_col(method_label):
    return f"{method_label} Present F1"


def _weighted_col(method_label):
    return f"Weighted {method_label} F1"


def _runtime_col(method_label):
    return f"{method_label} Runtime (seconds)"


def flatten_list(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item


def _merged_training_logs(training_data, timestep, window_size=2):
    if timestep not in training_data:
        return []

    if timestep == 0:
        return training_data[timestep]

    keys = sorted(training_data.keys())
    idx = keys.index(timestep)
    start = max(0, idx - window_size + 1)
    selected = keys[start: idx + 1]
    return list(flatten_list([training_data[k] for k in selected]))


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


def _evaluate_stateless_baseline(summary_method, method_name, KG, t, K, training_data, test_data, weights):
    method_label = _baseline_label(method_name)
    norm_w = _normalize_weights(weights)
    merged_logs = _merged_training_logs(training_data, t)

    t0 = time.time()
    summary = summary_method(KG, K, merged_logs)
    run_time = time.time() - t0

    total_F1, _, _ = total_query_log_metrics(summary, test_data[t])
    present_row = {
        "Timestep": t,
        _present_col(method_label): total_F1,
        "Runtime (seconds)": run_time,
    }

    df_w = process_all_timesteps(
        training_set=test_data,
        summary=summary,
        i=t,
        test_weights=norm_w,
    )

    return present_row, df_w


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


    common_keys = top_k_nodes_dict.keys() & graphs.keys() & weights.keys() & test_data.keys()

    for key in sorted(common_keys):

        KG = graphs[key]
        top_k_nodes = {str(n): v for n, v in top_k_nodes_dict[key].items()}
        non_top_k_nodes = {str(n): v for n, v in global_pr_dict[key].items()}

        #eliminate top-k from global pr
        non_top_k_nodes =  {k:v for k,v in non_top_k_nodes.items() if k not in top_k_nodes}


        # -------- SUMMARY + RUNTIME --------
        t0 = time.time()

        S =  SummaryGraph(KG, top_k_nodes, non_top_k_nodes, max_triples)

        t1 = time.time()

        summary_runtime = t1 - t0
        summary_runtime_rows.append({"Timestep": key, "Runtime (summary)": summary_runtime})
        triples_no[key] = S.triple_count
        _log_sankofa_summary(key, S)


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
def run_stateless_baseline(summary_method, method_name, graphs, triples_no_mc, training_data, test_data, weights):
    rows = []
    weighted_tests = []

    for t in sorted(graphs.keys() & triples_no_mc.keys() & weights.keys()):
        KG = graphs[t]
        K = triples_no_mc[t]

        present_row, df_w = _evaluate_stateless_baseline(
            summary_method,
            method_name,
            KG,
            t,
            K,
            training_data,
            test_data,
            weights[t],
        )
        rows.append(present_row)
        weighted_tests.append(df_w)

    return rows, weighted_tests


def run_glimpse(summary_method, graphs, triples_no_mc, training_data, test_data, weights):
    return run_stateless_baseline(summary_method, "glimpse", graphs, triples_no_mc, training_data, test_data, weights)


# --------------------- process_summary_APEX  ---------------------
def run_apex(graphs, triples_no_mc, training_data, test_data, weights, method_name="apex"):
    apex_rows = []
    weighted_tests = []
    method_label = _baseline_label(method_name)

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

        apex_rows.append({"Timestep": t, _present_col(method_label): total_F1, "Runtime (seconds)": run_time})
        df_w = process_all_timesteps(training_set=test_data, summary=P, i=t, test_weights=_normalize_weights(test_w))
        weighted_tests.append(df_w)



    return apex_rows, weighted_tests


# --------------------- process_summary_ISUMMARY  ---------------------
def run_isummary(summary_method, graphs, triples_no_mc, training_data, test_data, weights):
    return run_stateless_baseline(summary_method, "isummary", graphs, triples_no_mc, training_data, test_data, weights)


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
    Orchestrates baseline runs and merges results.
    Returns: DataFrame (sota) with columns for each method.
    """

    method_names = _validated_method_names(methods)
    if not method_names:
        return pd.DataFrame()

    raw_results = {
        method_name: {"present": [], "weighted": []}
        for method_name in method_names
    }

    for method_name in method_names:
        summary_method = METHODS[method_name]
        method_label = _baseline_label(method_name)

        print(f"Summarizing for {method_label}")

        if method_name == "apex":
            rows, weighted = run_apex(
                graphs,
                triples_no_mc,
                training_data,
                test_data,
                weights,
                method_name=method_name,
            )
        else:
            rows, weighted = run_stateless_baseline(
                summary_method,
                method_name,
                graphs,
                triples_no_mc,
                training_data,
                test_data,
                weights,
            )

        raw_results[method_name]["present"].extend(rows)
        raw_results[method_name]["weighted"].extend(weighted)

 
    merged_results = {}
    for method_name, result in raw_results.items():
        present = result["present"]
        weighted = result["weighted"]
        if not present:   
            continue

        method_label = _baseline_label(method_name)

        merged = _merge_present_and_weighted(
            present,
            weighted,
            _present_col(method_label),
            _weighted_col(method_label)
        )

        if merged is None or merged.empty:
            continue


        merged.rename(columns={"Runtime (seconds)": _runtime_col(method_label)}, inplace=True)

        print(f"{method_label}\n", merged)
        merged.to_csv(f"{method_name}_results.csv", index=False)

        merged_results[method_name] = merged

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
    # """Load all snapshots from the specified directory."""
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
    
    return G


# def generate_query_logs(graph, timestep, num_queries=200):
#     """Generate query logs for a given graph."""
#     topic_mids = random.sample(list(graph.triples_.keys()), k=5)
#     valid_topic_mids = [mid for mid in topic_mids if mid in graph.triples_]
#     if not valid_topic_mids:
#         print(f"No valid topic_mids found in KG for snapshot {timestep}")
#         return None

#     query_log = query_log_by_mids(
#         graph,
#         valid_topic_mids,
#         num_queries,
#         topic_dist=np.ones(len(valid_topic_mids)) / len(valid_topic_mids),
#         shuffle=True,
#         random_query_prob=0
#     )
#     return query_log


# def save_or_load_query_log(query_log_file, graph, timestep):
#     """Save or load query logs from a file."""

#     if os.path.exists(query_log_file):
#         with open(query_log_file, "rb") as f:
#             return pickle.load(f)
#     else:
#         query_log = generate_query_logs(graph, timestep)
#         with open(query_log_file, "wb") as f:
#             pickle.dump(query_log, f)
#         return query_log
    



def process_snapshots(subgraph_dir):
    """Process all snapshots and generate graphs, query logs, and training/test data."""
    graphs, training_data, test_data, queries_ = {}, {}, {}, {}

    files = os.listdir(subgraph_dir)
    snapshot_files = sorted(
    [f for f in files if f.startswith("snapshot_") and f.endswith(".nt.gz")],
    key=extract_snapshot_number
        )

    #print("Loading all subgraph snapshots...")

    for i, file_name in enumerate(snapshot_files):
        #file_name = "snapshot_1.nt.gz"
        path = os.path.join(subgraph_dir, file_name)
        KG = DBPedia(rdf_gz=path)
        KG.load()
        graphs[i] = KG
        #print(f"Loaded snapshot {file_name} as timestep {i}, size: {KG.number_of_triples()} triples ,node_size: {KG.number_of_entities()}")
        query_log_file = f"query_log_snapshot_{i}.pkl"
        query_log = save_or_load_query_log(query_log_file, KG, i)
        if not query_log:
            continue

        split_idx = int(0.8 * len(query_log))
        training_data[i] = query_log[:split_idx]
        test_data[i] = query_log[split_idx:]
        queries_[i] = _build_seed_node_list(training_data[i])
    
    action = "Loaded existing" if os.path.exists(query_log_file) else "Generated new"
    print(f"{action} query_logs")
    return graphs, training_data, test_data, queries_



###########################################################################################


def load_single_snapshot(subgraph_dir, file_name):
    """
    Load exactly one snapshot and return the KG.
    """
    path = os.path.join(subgraph_dir, file_name)
    KG = DBPedia(rdf_gz=path)
    KG.load()
    return KG


import pickle
from pathlib import Path

def prepare_queries_and_splits_(subgraph_dir):
    """
    Use ONE graph snapshot,
    but stream ALL existing query logs as timesteps.
    No query generation allowed.
    """

    training_data, test_data, queries_ = {}, {}, {}

    # --------------------------------------------------
    # 1️⃣ Get available graph snapshots
    # --------------------------------------------------
    files = os.listdir(subgraph_dir)
    snapshot_files_all = sorted(
        [f for f in files if f.startswith("snapshot_") and f.endswith(".nt.gz")],
        key=extract_snapshot_number
    )

    if not snapshot_files_all:
        raise ValueError("No graph snapshots found.")

    # 🔥 Always use the FIRST snapshot file
    single_snapshot = snapshot_files_all[0]
    #print(f"Using single graph snapshot: {single_snapshot}")

    # --------------------------------------------------
    # 2️⃣ Load ALL existing query logs
    # --------------------------------------------------
    QUERY_LOG_SOURCE = Path("/Users/au735195/Downloads/ExploratoryQueryingSessions-main/sessions/querylog/query_logsfinal_snapshots")#("/Users/au735195/Downloads/ExploratoryQueryingSessions-main/sessions/query_logs_snapshots_test")

    query_files = list(
        QUERY_LOG_SOURCE.glob("query_log_snapshot_*.pkl")
    )

    if not query_files:
        raise ValueError("No query snapshot files found.")
    

    query_files.sort(
    key=lambda p: int(p.stem.split("_")[-1])
    )
    

    query_files = sorted(
        query_files,
        key=lambda p: int(re.search(r"query_log_snapshot_(\d+)\.pkl", p.name).group(1))
    )

    # We fake snapshot_files so main() still works
    snapshot_files = [single_snapshot] * len(query_files)
    # snapshot_files = sorted(
    #     [f for f in files if f.startswith("snapshot_") and f.endswith(".nt.gz")],
    #     key=extract_snapshot_number
    # )

    

    #for i, qfile in enumerate(query_files):
    for i, (qfile, file_name) in enumerate(zip(query_files, snapshot_files)):

        #print(f"Loading queries from {qfile.name}")
        print(f"Using single graph snapshot: {file_name}, and query log: {qfile.name} for timestep {i}")

        # Load graph (same one every time)
        path = os.path.join(subgraph_dir, file_name)
        KG = DBPedia(rdf_gz=path)
        KG.load()

        with open(qfile, "rb") as f:
            query_log = pickle.load(f)

        if not query_log:
            del KG
            continue

        split_idx = int(0.8 * len(query_log))
        training_data[i] = query_log[:split_idx]
        test_data[i] = query_log[split_idx:]
        queries_[i] = _build_seed_node_list(training_data[i])

        del KG  # Free memory immediately

    print("Prepared query logs (single-graph, multi-query mode).")

    return training_data, test_data, queries_, snapshot_files


def prepare_queries_and_splits__(subgraph_dir):
    """
    Generate training/test splits and query lists
    WITHOUT storing all graphs in memory.
    """
    training_data, test_data, queries_ = {}, {}, {}

    files = os.listdir(subgraph_dir)
    snapshot_files = sorted(
        [f for f in files if f.startswith("snapshot_") and f.endswith(".nt.gz")],
        key=extract_snapshot_number
    )

    for i, file_name in enumerate(snapshot_files):
        path = os.path.join(subgraph_dir, file_name)

        KG = DBPedia(rdf_gz=path)
        KG.load()

        query_log_file = f"query_log_snapshot_{i}.pkl"
        query_log = save_or_load_query_log(query_log_file, KG, i)

        if not query_log:
            continue

        split_idx = int(0.8 * len(query_log))
        training_data[i] = query_log[:split_idx]
        test_data[i] = query_log[split_idx:]
        queries_[i] = _build_seed_node_list(training_data[i])

        # Free memory immediately
        del KG

    print("Prepared query logs (streaming mode).")

    return training_data, test_data, queries_, snapshot_files




def process_summary_sankofa_streaming(
    top_k_nodes_dict,
    snapshot_files,
    subgraph_dir,
    label,
    weights,
    global_pr_dict,
    test_data,
    walks_runtime=None,
    max_triples=None,
    graphs_by_timestep=None,
    verbose=True,
):

    triples_no = {}
    future_rows = []
    weighted_tests = []
    summary_runtime_rows = []

    # Only intersect lightweight dicts
    common_keys = (
        top_k_nodes_dict.keys()
        & weights.keys()
        & test_data.keys()
    )

    for key in sorted(common_keys):

        if verbose:
            progress_write(f"Evaluating timestep {key}")

        KG = None
        if graphs_by_timestep is not None:
            KG = graphs_by_timestep.get(key)

        if KG is None:
            file_name = snapshot_files[key]
            KG = load_single_snapshot(subgraph_dir, file_name)

        top_k_nodes = {str(n): v for n, v in top_k_nodes_dict[key].items()}
        non_top_k_nodes = {str(n): v for n, v in global_pr_dict[key].items()}
        non_top_k_nodes = {k: v for k, v in non_top_k_nodes.items() if k not in top_k_nodes}

        # -------- SUMMARY + RUNTIME --------
        t0 = time.time()
        S = SummaryGraph(KG, top_k_nodes, non_top_k_nodes, max_triples)
        t1 = time.time()

        summary_runtime = t1 - t0
        summary_runtime_rows.append({
            "Timestep": key,
            "Runtime (summary)": summary_runtime
        })

        triples_no[key] = S.triple_count
        _log_sankofa_summary(key, S)

        # -------- PRESENT TEST F1 --------
        log = test_data[key]
        total_F1, total_precision, total_recall = total_query_log_metrics(S, log)

        future_rows.append({
            "Timestep": key,
            "Future Test F1": total_F1
        })

        # -------- HISTORICAL TESTS --------
        test_weights = weights[key]
        total_w = sum(test_weights.values())
        test_weights_norm = {
            j: test_weights[j] / total_w for j in test_weights
        }

        df_weight = process_all_timesteps(
            training_set=test_data,
            summary=S,
            i=key,
            test_weights=test_weights_norm
        )

        weighted_tests.append(df_weight)

        # 🔥 IMPORTANT: FREE MEMORY
        del KG
        del S

    # -------- BUILD DATAFRAMES --------
    df_future = pd.DataFrame(future_rows)
    df_summary_rt = pd.DataFrame(summary_runtime_rows)

    if len(weighted_tests) > 0:
        df_weighted = pd.concat(weighted_tests, ignore_index=True)
        df_weighted = df_weighted.rename(columns={
            "Timestep Window": "Timestep",
            "Total F1": "Weighted Future Test F1"
        })
    else:
        df_weighted = pd.DataFrame(columns=[
            "Timestep", "Weighted Future Test F1"
        ])

    for df in (df_future, df_summary_rt, df_weighted):
        if not df.empty:
            df["Timestep"] = df["Timestep"].astype(int)

    if walks_runtime is None:
        walks_runtime = pd.DataFrame({
            "Timestep": sorted(common_keys),
            "Runtime (walks)": [0.0] * len(common_keys)
        })

    if "Runtime (walks)" not in walks_runtime.columns:
        walks_runtime["Runtime (walks)"] = 0.0

    # -------- MERGE --------
    merged = walks_runtime.merge(df_summary_rt, on="Timestep", how="right")
    merged = merged.merge(df_future, on="Timestep", how="left")
    merged = merged.merge(df_weighted, on="Timestep", how="left")

    merged["Weighted Future Test F1"] = merged["Weighted Future Test F1"].fillna(0.0)

    merged["Runtime (seconds)"] = (
        merged["Runtime (summary)"].fillna(0.0) +
        merged["Runtime (walks)"].fillna(0.0)
    )

    sankofa = merged[[
        "Timestep",
        "Runtime (seconds)",
        "Future Test F1",
        "Weighted Future Test F1"
    ]].copy()

    if verbose:
        progress_write(sankofa)

    return triples_no, sankofa


def evaluate_sankofa_timestep(
    timestep,
    KG,
    top_k_nodes,
    test_data,
    weights,
    global_pr,
    *,
    max_triples=None,
    walk_runtime=None,
):
    top_k_nodes = {str(n): v for n, v in top_k_nodes.items()}
    non_top_k_nodes = {str(n): v for n, v in global_pr.items()}
    non_top_k_nodes = {k: v for k, v in non_top_k_nodes.items() if k not in top_k_nodes}

    started = time.time()
    summary = SummaryGraph(KG, top_k_nodes, non_top_k_nodes, max_triples)
    summary_runtime = time.time() - started
    _log_sankofa_summary(timestep, summary)

    total_F1, _, _ = total_query_log_metrics(summary, test_data[timestep])

    df_weight = process_all_timesteps(
        training_set=test_data,
        summary=summary,
        i=timestep,
        test_weights=_normalize_weights(weights),
    )

    weighted_future_f1 = 0.0
    if df_weight is not None and not df_weight.empty and "Total F1" in df_weight.columns:
        weighted_future_f1 = float(df_weight["Total F1"].iloc[0])

    walk_seconds = 0.0 if walk_runtime is None else float(walk_runtime)
    sankofa_row = pd.DataFrame([
        {
            "Timestep": int(timestep),
            "Runtime (seconds)": summary_runtime + walk_seconds,
            "Future Test F1": total_F1,
            "Weighted Future Test F1": weighted_future_f1,
        }
    ])
    composition_row = pd.DataFrame([summary.composition_stats(timestep)])

    return summary.triple_count, sankofa_row, composition_row


class BaselineEngine:

    def __init__(self, methods):
        self.methods = _validated_method_names(methods)
        self.apex_state = {
            "H": None,
            "q": None,
            "e": None,
            "r": None,
            "csr": None,
            "index_list": None
        }

        self.present_rows = {method_name: [] for method_name in self.methods}

        self.weighted_rows = {method_name: [] for method_name in self.methods}

    def process_timestep(self, KG, t, K, training_data, test_data, weights, graph_delta=None, progress_prefix=None):

        phase_started = time.perf_counter()

        if progress_prefix is not None:
            progress_write(f"{progress_prefix}: running baselines ({len(self.methods)} total)...")

        for method_name in self.methods:
            method_started = time.perf_counter()
            method_label = _baseline_label(method_name)

            if progress_prefix is not None:
                progress_write(f"{progress_prefix}: baseline {method_label} started")

            if method_name == "apex":
                self._run_apex(method_name, KG, t, K, training_data, test_data, weights, graph_delta)
            else:
                self._run_stateless_baseline(method_name, KG, t, K, training_data, test_data, weights)

            runtime_seconds = self.present_rows[method_name][-1].get("Runtime (seconds)")
            method_runtime = (
                format_seconds(runtime_seconds)
                if runtime_seconds is not None
                else format_seconds(time.perf_counter() - method_started)
            )

            if progress_prefix is not None:
                progress_write(f"{progress_prefix}: baseline {method_label} finished dt={method_runtime}")

        if progress_prefix is not None:
            progress_write(
                f"{progress_prefix}: baselines complete in {format_seconds(time.perf_counter() - phase_started)}"
            )


    def _run_stateless_baseline(self, method_name, KG, t, K, training_data, test_data, weights):
        present_row, df_w = _evaluate_stateless_baseline(
            METHODS[method_name],
            method_name,
            KG,
            t,
            K,
            training_data,
            test_data,
            weights,
        )

        self.present_rows[method_name].append(present_row)
        self.weighted_rows[method_name].append(df_w)


    def _run_glimpse(self, KG, t, K, training_data, test_data, weights):
        self._run_stateless_baseline("glimpse", KG, t, K, training_data, test_data, weights)


    def _run_apex(self, method_name, KG, t, K, training_data, test_data, weights, graph_delta=None):
        method_label = _baseline_label(method_name)

        test_w = weights

        if t == 0:
            P, H, q, e, r, csr, index_list, total_F1, run_time = APEX(
                KG, K, training_data[t], test_data[t], t=t
            )
        else:
            H_prev, q_prev, e_prev, r_prev, csr_prev, index_prev = self._prepare_apex_state_for_graph(KG, graph_delta)
            try:
                P, H, q, e, r, csr, index_list, total_F1, run_time = APEX(
                    KG, K, training_data[t], test_data[t],
                    t=t,
                    H=H_prev,
                    q=q_prev,
                    e=e_prev,
                    r=r_prev,
                    csr_indirect_matrices=csr_prev,
                    index_list=index_prev
                )
            except Exception:
                P, H, q, e, r, csr, index_list, total_F1, run_time = APEX(
                    KG, K, training_data[t], test_data[t], t=t
                )

        self.apex_state.update({
            "H": H,
            "q": q,
            "e": e,
            "r": r,
            "csr": csr,
            "index_list": index_list
        })

        self.present_rows[method_name].append({
            "Timestep": t,
            _present_col(method_label): total_F1,
            "Runtime (seconds)": run_time
        })

        df_w = process_all_timesteps(
            training_set=test_data,
            summary=P,
            i=t,
            test_weights=_normalize_weights(test_w)
        )

        self.weighted_rows[method_name].append(df_w)


    def _prepare_apex_state_for_graph(self, KG, graph_delta):
        H_prev = self.apex_state["H"]
        q_prev = self.apex_state["q"]
        e_prev = self.apex_state["e"]
        r_prev = self.apex_state["r"]
        csr_prev = self.apex_state["csr"]
        index_prev = self.apex_state["index_list"]

        entity_count = KG.number_of_entities()
        relation_count = KG.number_of_relationships()

        if q_prev is not None and len(q_prev) < entity_count:
            q_prev = np.pad(q_prev, (0, entity_count - len(q_prev)))

        if e_prev is not None and len(e_prev) < entity_count:
            e_prev = np.pad(e_prev, (0, entity_count - len(e_prev)))

        if r_prev is not None and len(r_prev) < relation_count:
            r_prev = np.pad(r_prev, (0, relation_count - len(r_prev)))

        if graph_delta is not None:
            removed_triples = {
                (
                    KG.entity_id(triple[0]),
                    KG.relation_id(triple[1]),
                    KG.entity_id(triple[2]),
                )
                for triple in graph_delta.get("triples_removed", [])
            }

            if H_prev is not None:
                H_prev = dict(H_prev)
                for triple_id in removed_triples:
                    H_prev[triple_id] = 0.0

            if index_prev is not None:
                index_prev = [triple_id for triple_id in index_prev if triple_id not in removed_triples]

            csr_prev = None

        return H_prev, q_prev, e_prev, r_prev, csr_prev, index_prev


    def _run_isummary(self, KG, t, K, training_data, test_data, weights):
        self._run_stateless_baseline("isummary", KG, t, K, training_data, test_data, weights)

    def finalize(self):

        merged_results = []

        for method_name in self.methods:

            present = self.present_rows.get(method_name, [])
            weighted = self.weighted_rows.get(method_name, [])

            if not present:
                continue

            method_label = _baseline_label(method_name)

            merged = _merge_present_and_weighted(
                present,
                weighted,
                _present_col(method_label),
                _weighted_col(method_label)
            )

            if "Runtime (seconds)" in merged.columns:
                merged = merged.rename(columns={"Runtime (seconds)": _runtime_col(method_label)})

            merged_results.append(merged)

        if not merged_results:
            return pd.DataFrame()

        sota = merged_results[0]
        for df in merged_results[1:]:
            sota = pd.merge(sota, df, on="Timestep", how="inner")

        return sota
    



###############################################################NEW CHAPTER####################################################################
################################################################################################################################################
import random

def update_topics_fixed(prev_topics, graph, K=5, change_ratio=0.4, rng=None):
    """
    Fixed number of topics per timestep.
    40% change, 60% persist.
    Only keeps topics valid in current graph.
    """

    rng = random if rng is None else rng

    all_nodes = list(graph.triples_.keys())

    # First timestep
    if prev_topics is None:
        return rng.sample(all_nodes, k=min(K, len(all_nodes)))

    # Keep only topics still in graph
    valid_prev = [t for t in prev_topics if t in graph.triples_]

    # Number to keep
    num_keep_target = int((1 - change_ratio) * K)
    num_keep = min(num_keep_target, len(valid_prev))

    kept = rng.sample(valid_prev, k=num_keep) if num_keep > 0 else []

    # Fill remaining with new topics
    num_new = K - len(kept)
    candidates = list(set(all_nodes) - set(kept))

    if len(candidates) >= num_new:
        new_topics = rng.sample(candidates, k=num_new)
    else:
        new_topics = candidates

    topics = kept + new_topics

    # Ensure exactly K topics
    if len(topics) < K:
        extra_needed = K - len(topics)
        extra_candidates = list(set(all_nodes) - set(topics))
        if extra_candidates:
            topics += rng.sample(
                extra_candidates,
                k=min(extra_needed, len(extra_candidates))
            )

    topics = topics[:K]

    # Safety check
    assert all(t in graph.triples_ for t in topics)

    return topics



QUERY_LOGS_DIR_NAME = "query_logs"
QUERY_LOG_BUNDLE_METADATA_FILE = "bundle_metadata.json"
QUERY_LOG_TOPIC_COUNT = 5
QUERY_LOG_TOPIC_CHANGE_RATIO = 0.4
QUERY_LOG_NUM_QUERIES = 200
QUERY_LOG_SHUFFLE = True
QUERY_LOG_RANDOM_QUERY_PROB = 0.0


def _query_generation_settings():
    return {
        "topic_strategy": "update_topics_fixed",
        "topic_count": QUERY_LOG_TOPIC_COUNT,
        "topic_change_ratio": QUERY_LOG_TOPIC_CHANGE_RATIO,
        "query_strategy": "query_log_by_mids",
        "num_queries": QUERY_LOG_NUM_QUERIES,
        "shuffle": QUERY_LOG_SHUFFLE,
        "random_query_prob": QUERY_LOG_RANDOM_QUERY_PROB,
    }


def _format_query_setting_value(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return format(value, "g").replace("-", "m").replace(".", "p")
    return str(value).replace("-", "m").replace(".", "p")


def _query_bundle_prefix(settings):
    return (
        f"tc{settings['topic_count']}_"
        f"cr{_format_query_setting_value(settings['topic_change_ratio'])}_"
        f"q{settings['num_queries']}_"
        f"s{_format_query_setting_value(settings['shuffle'])}_"
        f"p{_format_query_setting_value(settings['random_query_prob'])}"
    )


def _query_bundle_name(settings, bundle_id):
    return f"{_query_bundle_prefix(settings)}_{bundle_id:03d}"


def _query_bundle_root(subgraph_dir):
    return Path(subgraph_dir) / QUERY_LOGS_DIR_NAME


def _query_bundle_dir(subgraph_dir, settings, bundle_id):
    return _query_bundle_root(subgraph_dir) / _query_bundle_name(settings, bundle_id)


def _query_log_filename(timestep):
    return f"query_log_snapshot_{timestep}.pkl"


def _query_log_path(bundle_dir, timestep):
    return bundle_dir / _query_log_filename(timestep)


def _query_bundle_metadata_path(bundle_dir):
    return bundle_dir / QUERY_LOG_BUNDLE_METADATA_FILE


def _snapshot_metadata(snapshot_path):
    stat = snapshot_path.stat()
    return {
        "name": snapshot_path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_query_bundle_metadata(bundle_dir):
    metadata_path = _query_bundle_metadata_path(bundle_dir)
    if not metadata_path.exists():
        return None

    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _query_bundle_id_from_name(bundle_name, settings):
    pattern = rf"{re.escape(_query_bundle_prefix(settings))}_(\d+)"
    match = re.fullmatch(pattern, bundle_name)
    if match is None:
        return None
    return int(match.group(1))


def _query_bundle_is_valid(bundle_dir, metadata, settings, snapshot_metadata):
    if not isinstance(metadata, dict):
        return False
    if metadata.get("generation_settings") != settings:
        return False
    if metadata.get("snapshot_files") != snapshot_metadata:
        return False
    if metadata.get("query_log_count") != len(snapshot_metadata):
        return False

    bundle_id = metadata.get("bundle_id")
    if not isinstance(bundle_id, int):
        return False

    for timestep in range(len(snapshot_metadata)):
        if not _query_log_path(bundle_dir, timestep).exists():
            return False

    return True


def _find_matching_query_bundles(subgraph_dir, settings, snapshot_files):
    bundle_root = _query_bundle_root(subgraph_dir)
    snapshot_paths = [Path(subgraph_dir) / file_name for file_name in snapshot_files]
    snapshot_metadata = [_snapshot_metadata(path) for path in snapshot_paths]
    matches = []

    if not bundle_root.exists():
        return matches

    for bundle_dir in bundle_root.iterdir():
        if not bundle_dir.is_dir():
            continue

        metadata = _load_query_bundle_metadata(bundle_dir)
        if not _query_bundle_is_valid(bundle_dir, metadata, settings, snapshot_metadata):
            continue

        matches.append(
            {
                "bundle_id": metadata["bundle_id"],
                "bundle_name": bundle_dir.name,
                "bundle_dir": bundle_dir,
                "status": "reused",
            }
        )

    return sorted(matches, key=lambda item: item["bundle_id"])


def _next_query_bundle_id(subgraph_dir, settings):
    bundle_root = _query_bundle_root(subgraph_dir)
    max_bundle_id = 0

    if not bundle_root.exists():
        return 1

    for bundle_dir in bundle_root.iterdir():
        if not bundle_dir.is_dir():
            continue
        bundle_id = _query_bundle_id_from_name(bundle_dir.name, settings)
        if bundle_id is not None:
            max_bundle_id = max(max_bundle_id, bundle_id)

    return max_bundle_id + 1


def _bundle_rngs(settings, bundle_id, snapshot_files):
    seed_material = json.dumps(
        {
            "bundle_id": bundle_id,
            "settings": settings,
            "snapshot_files": list(snapshot_files),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    python_seed = int.from_bytes(digest[:8], "big", signed=False)
    numpy_seed = int.from_bytes(digest[8:12], "big", signed=False)
    return random.Random(python_seed), np.random.RandomState(numpy_seed)


def _generate_query_bundle(subgraph_dir, snapshot_files, settings, bundle_id, *, replace_existing=False):
    return _generate_query_bundles(
        subgraph_dir,
        snapshot_files,
        settings,
        [bundle_id],
        replace_existing=replace_existing,
    )[bundle_id]


def _generate_query_bundles(subgraph_dir, snapshot_files, settings, bundle_ids, *, replace_existing=False):
    bundle_started = time.perf_counter()
    bundle_root = _query_bundle_root(subgraph_dir)
    bundle_root.mkdir(parents=True, exist_ok=True)

    requested_bundle_ids = sorted(set(bundle_ids))
    bundle_states = {}
    snapshot_paths = [Path(subgraph_dir) / file_name for file_name in snapshot_files]

    for bundle_id in requested_bundle_ids:
        bundle_name = _query_bundle_name(settings, bundle_id)
        bundle_dir = _query_bundle_dir(subgraph_dir, settings, bundle_id)

        if replace_existing and bundle_dir.exists():
            if bundle_dir.is_dir():
                shutil.rmtree(bundle_dir)
            else:
                bundle_dir.unlink()

        bundle_dir.mkdir(parents=True, exist_ok=False)
        rng, np_rng = _bundle_rngs(settings, bundle_id, snapshot_files)
        bundle_states[bundle_id] = {
            "bundle_id": bundle_id,
            "bundle_name": bundle_name,
            "bundle_dir": bundle_dir,
            "prev_topics": None,
            "total_queries": 0,
            "rng": rng,
            "np_rng": np_rng,
        }

    progress_write(
        f"Query logs: generating stored bundle(s) {requested_bundle_ids} under {bundle_root} "
        f"for {len(snapshot_paths)} snapshots (settings={_query_bundle_prefix(settings)})."
    )

    bundle_steps = progress(
        enumerate(snapshot_paths),
        total=len(snapshot_paths),
        desc="Generating query bundles",
        unit="snapshot",
        leave=False,
    )

    max_workers = max(1, min(len(bundle_states), os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for timestep, snapshot_path in bundle_steps:
            started = time.perf_counter()
            kg_load_started = time.perf_counter()
            KG = DBPedia(rdf_gz=str(snapshot_path))
            KG.load()
            kg_load_seconds = time.perf_counter() - kg_load_started

            bundle_futures = []
            for bundle_id, state in bundle_states.items():
                current_topics = update_topics_fixed(
                    state["prev_topics"],
                    KG,
                    K=settings["topic_count"],
                    change_ratio=settings["topic_change_ratio"],
                    rng=state["rng"],
                )
                state["prev_topics"] = current_topics
                future = executor.submit(
                    generate_query_logs,
                    KG,
                    timestep,
                    current_topics,
                    num_queries=settings["num_queries"],
                    shuffle=settings["shuffle"],
                    random_query_prob=settings["random_query_prob"],
                    rng=state["rng"],
                    np_rng=state["np_rng"],
                )
                bundle_futures.append((bundle_id, current_topics, future))

            timestep_queries = 0
            timestep_topics = 0
            for bundle_id, current_topics, future in bundle_futures:
                query_log = future.result()
                query_count = len(query_log) if query_log else 0
                timestep_queries += query_count
                timestep_topics += len(current_topics)

                state = bundle_states[bundle_id]
                state["total_queries"] += query_count
                with _query_log_path(state["bundle_dir"], timestep).open("wb") as handle:
                    pickle.dump(query_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

            bundle_steps.set_postfix(
                snapshot=snapshot_path.name,
                dt=format_seconds(time.perf_counter() - started),
                load=format_seconds(kg_load_seconds),
                bundles=len(bundle_states),
                topics=timestep_topics,
                queries=timestep_queries,
            )
            del KG

    snapshot_metadata = [_snapshot_metadata(path) for path in snapshot_paths]
    generated = {}
    for bundle_id, state in bundle_states.items():
        metadata = {
            "bundle_id": bundle_id,
            "bundle_name": state["bundle_name"],
            "generation_settings": settings,
            "snapshot_files": snapshot_metadata,
            "query_log_count": len(snapshot_paths),
        }

        with _query_bundle_metadata_path(state["bundle_dir"]).open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        progress_write(
            f"Query logs: bundle {state['bundle_name']} ready snapshots={len(snapshot_paths)} "
            f"queries={state['total_queries']:,} dt={format_seconds(time.perf_counter() - bundle_started)}"
        )
        generated[bundle_id] = {
            "bundle_id": bundle_id,
            "bundle_name": state["bundle_name"],
            "bundle_dir": state["bundle_dir"],
            "status": "generated",
        }

    return generated


def _ensure_query_bundles(subgraph_dir, snapshot_files, query_bundle_indices=None):
    settings = _query_generation_settings()
    if query_bundle_indices is None:
        target_indices = [1]
    else:
        target_indices = sorted(set(query_bundle_indices))
        if any(query_bundle_index < 1 for query_bundle_index in target_indices):
            raise ValueError("query_bundle_index must be >= 1")

    bundle_root = _query_bundle_root(subgraph_dir)
    progress_write(
        f"Query logs: resolving bundle(s) {target_indices} under {bundle_root} "
        f"for {len(snapshot_files)} snapshots (settings={_query_bundle_prefix(settings)})."
    )
    matches = _find_matching_query_bundles(subgraph_dir, settings, snapshot_files)
    matches_by_id = {item["bundle_id"]: item for item in matches}
    matching_bundle_ids = sorted(matches_by_id)

    if matches:
        progress_write(
            f"Query logs: found {len(matches)} matching stored bundle(s); "
            f"matching ids={matching_bundle_ids}."
        )
    else:
        progress_write("Query logs: no matching stored bundles found.")

    missing_ids = []
    for target_index in target_indices:
        if target_index in matches_by_id:
            continue

        bundle_dir = _query_bundle_dir(subgraph_dir, settings, target_index)
        bundle_state = "stale existing" if bundle_dir.exists() else "missing"
        progress_write(
            f"Query logs: requested bundle {target_index} is {bundle_state}; "
            f"queued for exact regeneration."
        )
        missing_ids.append(target_index)

    if missing_ids:
        matches_by_id.update(
            _generate_query_bundles(
                subgraph_dir,
                snapshot_files,
                settings,
                missing_ids,
                replace_existing=True,
            )
        )

    selected_bundles = [matches_by_id[target_index] for target_index in target_indices]
    reused_bundles = [bundle for bundle in selected_bundles if bundle["status"] == "reused"]
    if reused_bundles:
        progress_write(
            f"Query logs: using stored bundle(s) {[bundle['bundle_name'] for bundle in reused_bundles]}."
        )

    return selected_bundles


def _ensure_query_bundle(subgraph_dir, snapshot_files, query_bundle_index=None):
    return _ensure_query_bundles(
        subgraph_dir,
        snapshot_files,
        query_bundle_indices=None if query_bundle_index is None else [query_bundle_index],
    )[0]


def _load_query_splits_from_bundle(bundle_dir, snapshot_files):
    training_data, test_data, queries_ = {}, {}, {}

    load_steps = progress(
        enumerate(snapshot_files),
        total=len(snapshot_files),
        desc=f"Loading query bundle {bundle_dir.name}",
        unit="snapshot",
        leave=False,
    )
    for timestep, file_name in load_steps:
        with _query_log_path(bundle_dir, timestep).open("rb") as handle:
            query_log = pickle.load(handle)

        query_count = len(query_log) if query_log else 0

        if not query_log:
            load_steps.set_postfix(snapshot=file_name, queries=0, train=0, test=0)
            continue

        split_idx = int(0.8 * len(query_log))
        training_data[timestep] = query_log[:split_idx]
        test_data[timestep] = query_log[split_idx:]
        queries_[timestep] = _build_seed_node_list(training_data[timestep])
        load_steps.set_postfix(
            snapshot=file_name,
            queries=query_count,
            train=len(training_data[timestep]),
            test=len(test_data[timestep]),
        )

    return training_data, test_data, queries_


def generate_query_logs(graph, timestep, topic_mids, num_queries=200, *, shuffle=True,
    random_query_prob=0, rng=None, np_rng=None):
    """Generate query logs using given topics."""

    valid_topic_mids = [mid for mid in topic_mids if mid in graph.triples_]

    if not valid_topic_mids:
        progress_write(f"No valid topic_mids found in KG for snapshot {timestep}")
        return None

    query_log = query_log_by_mids(
        graph,
        valid_topic_mids,
        num_queries,
        topic_dist=np.ones(len(valid_topic_mids)) / len(valid_topic_mids),
        shuffle=shuffle,
        random_query_prob=random_query_prob,
        rng=rng,
        np_rng=np_rng,
    )

    return query_log


def save_or_load_query_log(query_log_file, graph, timestep, topic_mids=None):
    """Save or load query logs from a file."""

    # Load if exists (DO NOT regenerate)
    if os.path.exists(query_log_file):
        with open(query_log_file, "rb") as f:
            return pickle.load(f)

    if topic_mids is None:
        topic_mids = update_topics_fixed(
            prev_topics=None,
            graph=graph,
            K=QUERY_LOG_TOPIC_COUNT,
            change_ratio=QUERY_LOG_TOPIC_CHANGE_RATIO,
        )

    query_log = generate_query_logs(
        graph,
        timestep,
        topic_mids,
        num_queries=QUERY_LOG_NUM_QUERIES,
        shuffle=QUERY_LOG_SHUFFLE,
        random_query_prob=QUERY_LOG_RANDOM_QUERY_PROB,
    )

    with open(query_log_file, "wb") as f:
        pickle.dump(query_log, f)

    return query_log


def prepare_queries_and_splits(subgraph_dir, query_bundle_index=None):
    """
    Generate training/test splits and query lists from a reusable stored query bundle.
    """
    files = os.listdir(subgraph_dir)
    snapshot_files = sorted(
        [f for f in files if f.startswith("snapshot_") and f.endswith(".nt.gz")],
        key=extract_snapshot_number
    )

    if not snapshot_files:
        raise ValueError(f"No snapshot_*.nt.gz files found in {subgraph_dir}")

    selected_bundle = _ensure_query_bundle(
        subgraph_dir,
        snapshot_files,
        query_bundle_index=query_bundle_index,
    )
    training_data, test_data, queries_ = _load_query_splits_from_bundle(
        selected_bundle["bundle_dir"],
        snapshot_files,
    )

    progress_write(
        f"Prepared query logs from bundle {selected_bundle['bundle_name']} "
        f"({selected_bundle['status']})."
    )

    return training_data, test_data, queries_, snapshot_files


def prepare_query_bundles(subgraph_dir, query_bundle_indices):
    files = os.listdir(subgraph_dir)
    snapshot_files = sorted(
        [f for f in files if f.startswith("snapshot_") and f.endswith(".nt.gz")],
        key=extract_snapshot_number
    )

    if not snapshot_files:
        raise ValueError(f"No snapshot_*.nt.gz files found in {subgraph_dir}")

    selected_bundles = _ensure_query_bundles(
        subgraph_dir,
        snapshot_files,
        query_bundle_indices=query_bundle_indices,
    )
    progress_write(
        f"Prepared query bundle(s) {[bundle['bundle_name'] for bundle in selected_bundles]}."
    )

    return selected_bundles, snapshot_files
import os


DEFAULT_MAIN_RUNTIME_SETTINGS = {
    "num_walks": 20,
    "c": 2,
    "epsilon": 0.90,
    "decay_rate": 0.8,
    "window_size": 100,
    "budget_ratio": 0.05,
    "baseline_methods": [],#,"hhop_random", "hhop_random_outgoing", "ppr_frequency"],
    "log_changes_in_topk": False,
    "seed_mode": "topic_only",# "answers_fallback_topic",
    "eval_decay_rate": 0.8,
    "use_rdf2vec": False,
    "rdf2vec_embedding_weight": 0.5,
}

MAIN_RUNTIME_ENV_VARS = {
    "num_walks": "QADGOSIS_NUM_WALKS",
    "c": "QADGOSIS_C",
    "epsilon": "QADGOSIS_EPSILON",
    "decay_rate": "QADGOSIS_DECAY_RATE",
    "window_size": "QADGOSIS_WINDOW_SIZE",
    "budget_ratio": "QADGOSIS_BUDGET_RATIO",
    "baseline_methods": "QADGOSIS_BASELINE_METHODS",
    "log_changes_in_topk": "QADGOSIS_LOG_CHANGES_IN_TOPK",
    "seed_mode": "QADGOSIS_SEED_MODE",
    "eval_decay_rate": "QADGOSIS_EVAL_DECAY_RATE",
    "use_rdf2vec": "QADGOSIS_USE_RDF2VEC",
    "rdf2vec_embedding_weight": "QADGOSIS_RDF2VEC_WEIGHT",
}

DEFAULT_QUERY_GENERATION_SETTINGS = {
    "topic_count": 1,
    "topic_change_ratio": 0.1,
    "num_queries": 15,
    "shuffle": True,
    "random_query_prob": 0.0,
    "chain_len": 1,
    # Hard-query mode: answers not on shortest path, abrupt topic shifts.
    "hard_query": True,
    "min_shift_hops": 1,
    "min_chain_len": 1,
    "max_chain_len": 1,
}

QUERY_GENERATION_ENV_VARS = {
    "topic_count": "QADGOSIS_QUERY_TOPIC_COUNT",
    "topic_change_ratio": "QADGOSIS_QUERY_TOPIC_CHANGE_RATIO",
    "num_queries": "QADGOSIS_QUERY_NUM_QUERIES",
    "shuffle": "QADGOSIS_QUERY_SHUFFLE",
    "random_query_prob": "QADGOSIS_QUERY_RANDOM_QUERY_PROB",
    "chain_len": "QADGOSIS_QUERY_CHAIN_LEN",
    "hard_query": "QADGOSIS_QUERY_HARD",
    "min_shift_hops": "QADGOSIS_QUERY_MIN_SHIFT_HOPS",
    "min_chain_len": "QADGOSIS_QUERY_MIN_CHAIN_LEN",
    "max_chain_len": "QADGOSIS_QUERY_MAX_CHAIN_LEN",
}


def _parse_bool(raw_value):
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_string_list(raw_value):
    return [item for item in (part.strip() for part in raw_value.split(",")) if item]


def _resolve_scalar(env, env_var, default_value, parser):
    raw_value = env.get(env_var)
    if raw_value is None or raw_value == "":
        return default_value
    return parser(raw_value)


def resolve_main_runtime_settings(env=None):
    env = os.environ if env is None else env
    return {
        "num_walks": _resolve_scalar(
            env,
            MAIN_RUNTIME_ENV_VARS["num_walks"],
            DEFAULT_MAIN_RUNTIME_SETTINGS["num_walks"],
            int,
        ),
        "c": _resolve_scalar(
            env,
            MAIN_RUNTIME_ENV_VARS["c"],
            DEFAULT_MAIN_RUNTIME_SETTINGS["c"],
            int,
        ),
        "epsilon": _resolve_scalar(
            env,
            MAIN_RUNTIME_ENV_VARS["epsilon"],
            DEFAULT_MAIN_RUNTIME_SETTINGS["epsilon"],
            float,
        ),
        "decay_rate": _resolve_scalar(
            env,
            MAIN_RUNTIME_ENV_VARS["decay_rate"],
            DEFAULT_MAIN_RUNTIME_SETTINGS["decay_rate"],
            float,
        ),
        "window_size": _resolve_scalar(
            env,
            MAIN_RUNTIME_ENV_VARS["window_size"],
            DEFAULT_MAIN_RUNTIME_SETTINGS["window_size"],
            int,
        ),
        "budget_ratio": _resolve_scalar(
            env,
            MAIN_RUNTIME_ENV_VARS["budget_ratio"],
            DEFAULT_MAIN_RUNTIME_SETTINGS["budget_ratio"],
            float,
        ),
        "baseline_methods": _resolve_scalar(
            env,
            MAIN_RUNTIME_ENV_VARS["baseline_methods"],
            list(DEFAULT_MAIN_RUNTIME_SETTINGS["baseline_methods"]),
            _parse_string_list,
        ),
        "log_changes_in_topk": _resolve_scalar(
            env,
            MAIN_RUNTIME_ENV_VARS["log_changes_in_topk"],
            DEFAULT_MAIN_RUNTIME_SETTINGS["log_changes_in_topk"],
            _parse_bool,
        ),
        "seed_mode": _resolve_scalar(
            env,
            MAIN_RUNTIME_ENV_VARS["seed_mode"],
            DEFAULT_MAIN_RUNTIME_SETTINGS["seed_mode"],
            str,
        ),
        "eval_decay_rate": _resolve_scalar(
            env,
            MAIN_RUNTIME_ENV_VARS["eval_decay_rate"],
            DEFAULT_MAIN_RUNTIME_SETTINGS["eval_decay_rate"],
            float,
        ),
        "use_rdf2vec": _resolve_scalar(
            env,
            MAIN_RUNTIME_ENV_VARS["use_rdf2vec"],
            DEFAULT_MAIN_RUNTIME_SETTINGS["use_rdf2vec"],
            _parse_bool,
        ),
        "rdf2vec_embedding_weight": _resolve_scalar(
            env,
            MAIN_RUNTIME_ENV_VARS["rdf2vec_embedding_weight"],
            DEFAULT_MAIN_RUNTIME_SETTINGS["rdf2vec_embedding_weight"],
            float,
        ),
    }


def resolve_query_generation_settings(env=None):
    env = os.environ if env is None else env
    return {
        "topic_count": _resolve_scalar(
            env,
            QUERY_GENERATION_ENV_VARS["topic_count"],
            DEFAULT_QUERY_GENERATION_SETTINGS["topic_count"],
            int,
        ),
        "topic_change_ratio": _resolve_scalar(
            env,
            QUERY_GENERATION_ENV_VARS["topic_change_ratio"],
            DEFAULT_QUERY_GENERATION_SETTINGS["topic_change_ratio"],
            float,
        ),
        "num_queries": _resolve_scalar(
            env,
            QUERY_GENERATION_ENV_VARS["num_queries"],
            DEFAULT_QUERY_GENERATION_SETTINGS["num_queries"],
            int,
        ),
        "shuffle": _resolve_scalar(
            env,
            QUERY_GENERATION_ENV_VARS["shuffle"],
            DEFAULT_QUERY_GENERATION_SETTINGS["shuffle"],
            _parse_bool,
        ),
        "random_query_prob": _resolve_scalar(
            env,
            QUERY_GENERATION_ENV_VARS["random_query_prob"],
            DEFAULT_QUERY_GENERATION_SETTINGS["random_query_prob"],
            float,
        ),
        "chain_len": _resolve_scalar(
            env,
            QUERY_GENERATION_ENV_VARS["chain_len"],
            DEFAULT_QUERY_GENERATION_SETTINGS["chain_len"],
            int,
        ),
        "hard_query": _resolve_scalar(
            env,
            QUERY_GENERATION_ENV_VARS["hard_query"],
            DEFAULT_QUERY_GENERATION_SETTINGS["hard_query"],
            _parse_bool,
        ),
        "min_shift_hops": _resolve_scalar(
            env,
            QUERY_GENERATION_ENV_VARS["min_shift_hops"],
            DEFAULT_QUERY_GENERATION_SETTINGS["min_shift_hops"],
            int,
        ),
        "min_chain_len": _resolve_scalar(
            env,
            QUERY_GENERATION_ENV_VARS["min_chain_len"],
            DEFAULT_QUERY_GENERATION_SETTINGS["min_chain_len"],
            int,
        ),
        "max_chain_len": _resolve_scalar(
            env,
            QUERY_GENERATION_ENV_VARS["max_chain_len"],
            DEFAULT_QUERY_GENERATION_SETTINGS["max_chain_len"],
            int,
        ),
    }


def main_runtime_settings_to_env(settings):
    return {
        MAIN_RUNTIME_ENV_VARS["num_walks"]: str(settings["num_walks"]),
        MAIN_RUNTIME_ENV_VARS["c"]: str(settings["c"]),
        MAIN_RUNTIME_ENV_VARS["epsilon"]: str(settings["epsilon"]),
        MAIN_RUNTIME_ENV_VARS["decay_rate"]: str(settings["decay_rate"]),
        MAIN_RUNTIME_ENV_VARS["window_size"]: str(settings["window_size"]),
        MAIN_RUNTIME_ENV_VARS["budget_ratio"]: str(settings["budget_ratio"]),
        MAIN_RUNTIME_ENV_VARS["baseline_methods"]: ",".join(settings["baseline_methods"]),
        MAIN_RUNTIME_ENV_VARS["log_changes_in_topk"]: "1" if settings["log_changes_in_topk"] else "0",
        MAIN_RUNTIME_ENV_VARS["seed_mode"]: settings["seed_mode"],
        MAIN_RUNTIME_ENV_VARS["eval_decay_rate"]: str(settings["eval_decay_rate"]),
    }


def query_generation_settings_to_env(settings):
    return {
        QUERY_GENERATION_ENV_VARS["topic_count"]: str(settings["topic_count"]),
        QUERY_GENERATION_ENV_VARS["topic_change_ratio"]: str(settings["topic_change_ratio"]),
        QUERY_GENERATION_ENV_VARS["num_queries"]: str(settings["num_queries"]),
        QUERY_GENERATION_ENV_VARS["shuffle"]: "1" if settings["shuffle"] else "0",
        QUERY_GENERATION_ENV_VARS["random_query_prob"]: str(settings["random_query_prob"]),
        QUERY_GENERATION_ENV_VARS["chain_len"]: str(settings["chain_len"]),
        QUERY_GENERATION_ENV_VARS["hard_query"]: "1" if settings["hard_query"] else "0",
        QUERY_GENERATION_ENV_VARS["min_shift_hops"]: str(settings["min_shift_hops"]),
        QUERY_GENERATION_ENV_VARS["min_chain_len"]: str(settings["min_chain_len"]),
        QUERY_GENERATION_ENV_VARS["max_chain_len"]: str(settings["max_chain_len"]),
    }

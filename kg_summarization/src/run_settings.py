import os


DEFAULT_MAIN_RUNTIME_SETTINGS = {
    "num_walks": 20,
    "c": 2,
    "epsilon": 0.90,
    "decay_rate": 0.8,
    "window_size": 3,
    "budget_ratio": 0.00001,
    "baseline_methods": [],#["glimpse","apex","isummary"],#, "hhop_random", "hhop_random_outgoing", "ppr_frequency"],
    "log_changes_in_topk": False,
    "seed_mode": "topic_only", #answers_fallback_topic
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
}

DEFAULT_QUERY_GENERATION_SETTINGS = {
    "topic_count": 5,
    "topic_change_ratio": 0.4,
    "num_queries": 200,
    "shuffle": True,
    "random_query_prob": 0.1,
}

QUERY_GENERATION_ENV_VARS = {
    "topic_count": "QADGOSIS_QUERY_TOPIC_COUNT",
    "topic_change_ratio": "QADGOSIS_QUERY_TOPIC_CHANGE_RATIO",
    "num_queries": "QADGOSIS_QUERY_NUM_QUERIES",
    "shuffle": "QADGOSIS_QUERY_SHUFFLE",
    "random_query_prob": "QADGOSIS_QUERY_RANDOM_QUERY_PROB",
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
    }


def query_generation_settings_to_env(settings):
    return {
        QUERY_GENERATION_ENV_VARS["topic_count"]: str(settings["topic_count"]),
        QUERY_GENERATION_ENV_VARS["topic_change_ratio"]: str(settings["topic_change_ratio"]),
        QUERY_GENERATION_ENV_VARS["num_queries"]: str(settings["num_queries"]),
        QUERY_GENERATION_ENV_VARS["shuffle"]: "1" if settings["shuffle"] else "0",
        QUERY_GENERATION_ENV_VARS["random_query_prob"]: str(settings["random_query_prob"]),
    }

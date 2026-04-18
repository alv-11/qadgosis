import numpy as np


def extract_query_seed_entities(query, seed_mode="topic_only"):
    """
    Return the entities used to seed query-driven baselines.
    """
    parse = query.get("Parse", {}) if isinstance(query, dict) else {}

    if seed_mode == "answers_fallback_topic":
        answers = []
        for answer in parse.get("Answers", []):
            if not isinstance(answer, dict):
                continue
            answer_arg = answer.get("AnswerArgument")
            if answer_arg is not None:
                answers.append(answer_arg)

        if answers:
            return answers

    if seed_mode != "topic_only" and seed_mode != "answers_fallback_topic":
        raise ValueError(f"Unsupported seed_mode: {seed_mode}")

    topic_entity = parse.get("TopicEntityMid")
    return [topic_entity] if topic_entity is not None else []


def query_vector(KG, query_log, seed_mode="topic_only"):
    """
    :param KG: KnowledgeGraph
    :param query_log: list of queries in dict format
    :param seed_mode: how to extract seed entities from each query
    :return x: query vector (n_entities,)
    """
    x = np.zeros(KG.number_of_entities())

    if seed_mode == "topic_only": #old code, only topic entity as seed
        for query in query_log:
            parse = query["Parse"]
            try:
                topic_eid = KG.entity_id(parse["TopicEntityMid"])
            except KeyError:
                continue
            x[topic_eid] += 1
        return x

    for query in query_log:
        for entity in extract_query_seed_entities(query, seed_mode=seed_mode):
            try:
                entity_id = KG.entity_id(entity)
            except KeyError:
                continue
            x[entity_id] += 1
    return x


def random_walk_with_restart(M, x, c=0.15, power=1):
    """
    :param M: scipy sparse transition matrix
    :param x: np.array (n_entities,) seed initializations
    :param c: float in [0, 1], optional restart prob
    :param power: number of terms in Taylor expansion
    :return r: np.array (n_entities,) random walk vector

    Approximates the matrix inverse using the Taylor expansion:
        (I - M)^-1 = I + M + M^2 + M^3 ...
    """
    q = c * np.copy(x)
    r = np.copy(q) # result vector

    if np.sum(r) == 0:
        return r

    for _ in range(power):
        q = (1 - c) * M * q
        r += q
        total = np.sum(r)
        if total != 0:
            r /= total
    return r

import numpy as np
import scipy.sparse as sp
from .base import KnowledgeGraph
from time import time
from .metrics import total_query_log_metrics, average_query_log_metrics
import logging
import pdb
import random
import warnings
from src.progress import format_seconds, progress, progress_write
DETAILED_LOGGING = True
DETAILED_PRINTING = False
MORE_DETAIL = False
GAMMA = 0.5

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=RuntimeWarning)
#     mean_val = np.mean(arr)

class Summary(KnowledgeGraph):

    def __init__(self, KG):
        """
        :param KG: KnowledgeGraph
        """
        super().__init__()
        self.parent_ = KG

    def parent(self):
        return self.parent_



    def fill(self, triples, k):
        """
        :param triples: triples to add to summary
        :param k: limit
        """
        for triple in triples:
            if self.number_of_triples() >= k:
                return
            self.add_triple(triple)

    # for better efficiency

    def add_entity(self, topic_mid):
        for entity in self.entities().copy():
            if self.parent().csr_matrix()[self.parent().entity_id_[entity], self.parent().entity_id_[topic_mid]] != 0: # exists a triple
                # find the relationship
                for r in self.parent().entities_to_relation_[entity][topic_mid]:
                    self.add_triple((entity, r, topic_mid))

            if self.parent().csr_matrix_T()[self.parent().entity_id_[entity], self.parent().entity_id_[topic_mid]] != 0: # exists a triple
                for r in self.parent().entities_to_relation_[topic_mid][entity]:
                    self.add_triple((topic_mid, r, entity))

        # in case self.entities() is empty
        self._ensure_entity(topic_mid)

    
    def add_random_triple(self):
        e1 = random.choice(list(self.parent().triples_.keys()))
        r = random.choice(list(self.parent()[e1].keys()))
        e2 = random.choice(list(self.parent()[e1][r]))
        self.add_triple((e1, r, e2))
        

# for debugging
def get_heat(heat, indices):
    h = []
    for i in indices:
        h.append(heat[i])
    return h



def Heat_Diffuse(heat, KG, query, d, csr_indirect_matrices):  # adj matrix and heat
    q = np.zeros(KG.number_of_entities())
    parse = query['Parse']
    topic_mid = parse['TopicEntityMid']
    topic_eid = KG.entity_id(parse['TopicEntityMid'])
    relation = parse['InferentialChain'][0]

    # Validate topic_mid before accessing
    if topic_mid not in KG.triples_:
        print(f"Warning: topic_mid '{topic_mid}' not found in KG.triples_")
        return heat, []

    try:
        answers = KG[topic_mid][relation]
    except KeyError:
        print(f"KeyError: relation '{relation}' not found for topic_mid '{topic_mid}'")
        return heat, []

    q[topic_eid] += 1
    answers_rid = np.zeros(len(answers), dtype=int)
    for i, item in enumerate(answers):
        answers_rid[i] = KG.entity_id(item)
        q[answers_rid[i]] += 1 / len(answers)

    # diffuse through edges (indirect)
    # TODO: Diffuse should be according to number of neighbors
    r = q.copy()
    for i in range(d):
        q = csr_indirect_matrices[i+1] * q
        r += q
    changed_index_list = []
    for i in range(len(r)):
        if r[i] != 0 and r[i] > 1e-3:
            changed_index_list.append(i)
        else:
            r[i] = 0
    r += heat

    return r, changed_index_list


def heat_decay_array(h, gamma, threshold = 1e-3):
    h = h * gamma
    for i in range(len(h)):
        if h[i] < threshold:
            h[i] = 0


def binary_find(index_list, changed_index, H):
    changed_value = H[changed_index]
    low = 0
    high = len(index_list) - 1
    mid = 0
    while high >= low:
        mid = (high + low)//2
        if H[index_list[mid]] < changed_value:
            high = mid - 1
        elif H[index_list[mid]] > changed_value:
            low = mid + 1
        else:
            if (index_list[mid] == changed_index):
                return mid
            else:
                mid_right = mid + 1
                mid_left = mid - 1
                while mid_right < len(index_list) and H[index_list[mid_right]] == changed_value:
                    if index_list[mid_right] == changed_index:
                        return mid_right
                    mid_right += 1
                while mid_left >= 0 and H[index_list[mid_left]] == changed_value:
                    if index_list[mid_left] == changed_index:
                        return mid_left
                    mid_left -= 1
    return -1
    



def binary_insert(index_list, changed_index, H):

    changed_value = H[changed_index]
    low = 0
    high = len(index_list) - 1

    if high == -1: # empty list
        index_list.append(changed_index)
        return
    
    # check side case
    if changed_value >= H[index_list[0]]:
        index_list.insert(0, changed_index)
        return
    if changed_value <= H[index_list[len(index_list) - 1]]:
        index_list.insert(len(index_list), changed_index)
        return 
    
    while high >= low + 2:
        mid = (high + low)//2
        if H[index_list[mid]] == changed_value:
            index_list.insert(mid, changed_index)
            return
        if H[index_list[mid]] < changed_value:
            high = mid
        else:
            low = mid

    #  low <= high <= low + 1
    index_list.insert(high, changed_index)


# not finished yet
def incremental_sort(index_list, changed_index_list, H_old, H_new):   # changed_index_list and prev_values are 1-1 corresponse
    # delete the changed elements
    positions = []
    # pdb.set_trace()
    for index in changed_index_list:
        pos = binary_find(index_list, index, H_old)
        if pos != -1:
            positions.append(pos)
    # pdb.set_trace()
    for index_del in sorted(positions, reverse=True):
        del index_list[index_del]

    # re-insert the changed elements
    for index in changed_index_list:
        if H_new[index] != 0:
            binary_insert(index_list, index, H_new)

    


def construct_naive(KG, K, topic_eids):
    if MORE_DETAIL:
        print('construct')
    P = Summary(KG)
    topic_mids = []
    # turn eid into mid
    for topic_eid in topic_eids:
        topic_mids.append(KG.id_entity_[topic_eid])
    i = 0
    while P.number_of_triples() < K:
        P.add_entity(topic_mids[i])
        i += 1
        if i == len(topic_eids):
            while P.number_of_triples() < K:
                # P.add_entity(KG.id_entity(np.random.randint(KG.number_of_entities())))
                P.add_random_triple()
            break
    if MORE_DETAIL:
        print('contruct end')

    return P

    
def construct_complete(KG, K, index_list):
    if MORE_DETAIL:
        print('construct start')
    P = Summary(KG)
    need_random = 1
    # i = 0
    # if len(index_list) == 0:
    #     print('here')
    #     need = K - P.number_of_triples()
    #     triple_to_add = random.sample(KG.triples(), k = need)
    #     for triple in triple_to_add:
    #         P.add_triple(triple)
    #     return P
    # while P.number_of_triples() < K:
    #     triple_to_add = (KG.id_entity(index_list[i][0]), KG.id_relation(index_list[i][1]), KG.id_entity(index_list[i][2]))
    #     P.add_triple(triple_to_add)
    #     i += 1
    #     if i == len(index_list):
    #         need = K - P.number_of_triples()
    #         triple_to_add = random.sample(KG.triples(), k = need)
    #         for triple in triple_to_add:
    #             P.add_triple(triple)
    #         break
    for i in range(len(index_list)):
        triple_to_add = (KG.id_entity(index_list[i][0]), KG.id_relation(index_list[i][1]), KG.id_entity(index_list[i][2]))
        P.add_triple(triple_to_add)
        if P.number_of_triples() >= K:
            need_random = 0
            break

    if need_random:
        need = K - P.number_of_triples()
        #triple_to_add = random.sample(KG.triples(), k = need)
        triple_to_add = random.sample(sorted(KG.triples()), k=need)
        for triple in triple_to_add:
            P.add_triple(triple)
        

    if MORE_DETAIL:
        print('construct end')
    return P


def _build_csr_indirect_matrices(KG, diameter, alpha):
    csr_indirect_matrices = {0: sp.eye(KG.number_of_entities())}
    indirect_heat = KG.csr_matrix_indirect_heat()
    for i in range(diameter):
        csr_indirect_matrices[i + 1] = alpha * indirect_heat * csr_indirect_matrices[i]
    return csr_indirect_matrices


def APEX_naive(KG, K, query_log, query_num_per_test=1, gamma=GAMMA, diameter = 1, alpha=0.3):
    print('APEX-N')
    update_time_list = []
    acc_list = []
    t = 0
    num_queries_total = len(query_log)
    csr_indirect_matrices = {}
    csr_indirect_matrices[0] = sp.eye(KG.number_of_entities())
    print('calculating heat matrices')
    for i in range(diameter):
        csr_indirect_matrices[i+1] = alpha * KG.csr_matrix_indirect_heat() * csr_indirect_matrices[i]
    # initialize heat

    print('calculating heat end')
    initial_heat = np.zeros(KG.number_of_entities())
    index_list = []
    first_q = query_log[0]
    heat, _ = Heat_Diffuse(initial_heat, KG, first_q, diameter, csr_indirect_matrices)
    sorted_heat = np.sort(heat)[::-1]
    sorted_args = np.argsort(heat)[::-1]
    i = 0
    while sorted_heat[i] > 0:
        index_list.append(sorted_args[i])
        i += 1
    P = construct_naive(KG, K, topic_eids = index_list)
    # test for initial graph
    if DETAILED_LOGGING:
        logging.info('Initial Test')
    total_F1, total_precision, total_recall = total_query_log_metrics(P, query_log[1: 1+query_num_per_test])
    if DETAILED_LOGGING:
        logging.info('\t  Total F1/precision/recall')
        logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
            total_F1, total_precision, total_recall))
    
    avg_F1, avg_precision, avg_recall = average_query_log_metrics(P, query_log[1: 1+query_num_per_test])
    if DETAILED_LOGGING:
        logging.info('\t  Average F1/precision/recall')
        logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
            avg_F1, avg_precision, avg_recall))

    # update phase
    t += 1
    while t < num_queries_total - query_num_per_test:
        print("Adapting for timestamp: ", t)
        t0 = time()
        heat = heat * gamma
        heat_new, changed_index_list = Heat_Diffuse(heat, KG, query_log[t], diameter, csr_indirect_matrices)
        for i in range(len(heat_new)):
            if heat_new[i] > 0 and heat_new[i] < 1e-4:
                heat_new[i] = 0

        # pdb.set_trace()
        if MORE_DETAIL:
            print('incremental')
        if DETAILED_PRINTING:
            print("Length Index List: ", len(index_list))
            print("Length Changed:", len(changed_index_list))
        print('incremental')
        incremental_sort(index_list, changed_index_list, heat, heat_new)
        # print('incremental end')
        new_index_list = []
        for i in range(len(index_list)):
            if heat_new[index_list[i]] != 0:
                new_index_list.append(index_list[i])
            else:
                break
        index_list = new_index_list
        # pdb.set_trace()
        heat = heat_new
        print('construct')
        P = construct_naive(KG, K, topic_eids= index_list)
        update_time = time() - t0
        if DETAILED_PRINTING:
            print("Length Index List: ", len(index_list))
            print("Length Changed:", len(changed_index_list))
        update_time_list.append(update_time)

        if DETAILED_LOGGING:
            logging.info('\t  Adapting for time: {}'.format(t))
            logging.info('\t  Time: {:.2f} seconds'.format(update_time))

        # Evaluate question answering on the testing queries
        total_F1, total_precision, total_recall = total_query_log_metrics(P, query_log[t+1: t+1+query_num_per_test])
        if DETAILED_LOGGING:
            logging.info('\t  Total F1/precision/recall')
            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                total_F1, total_precision, total_recall))
        
        avg_F1, avg_precision, avg_recall = average_query_log_metrics(P, query_log[t+1: t+1+query_num_per_test])
        if DETAILED_LOGGING:
            logging.info('\t  Average F1/precision/recall')
            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                avg_F1, avg_precision, avg_recall))
        
        acc_list.append(avg_F1)

        
        t += 1     



    logging.info('\t  Ave Time on Each Training Log: {:.2f} seconds'.format(np.mean(update_time_list)))
    logging.info('\t  Ave Ave F1 on Each Training Log: {:.2f}'.format(np.mean(acc_list)))


    np.save(KG.user_dir + '/acc_list_apex_n.npy', acc_list)
    np.save(KG.user_dir + '/time_list_apex_n.npy', update_time_list)

    
    # pdb.set_trace()
    return acc_list, update_time_list









def APEX(KG, K, training, query_log, query_num_per_test=1, gamma=GAMMA, diameter=1, alpha=0.3, t=0, H=None, q=None, e=None, r=None, csr_indirect_matrices=None, index_list=None):
    smoothing = 0.5
    update_time_list = []
    acc_list = []

    #apex = []

    if t == 0:
        # Initialization phase (unchanged)
        t0 = time()
        num_queries_total = len(query_log)
        csr_indirect_matrices = _build_csr_indirect_matrices(KG, diameter, alpha)

        # Initial state
        index_list = []
        q = np.zeros(KG.number_of_entities())
        e = np.zeros(KG.number_of_entities())
        r = np.zeros(KG.number_of_relationships())

        # Process all queries in training
        for query in training:
            parse = query['Parse']
            topic_mid = parse['TopicEntityMid']
            topic_eid = KG.entity_id(parse['TopicEntityMid'])
            #if topic_eid is None:
            #    print(f"Skipping query: TopicEntityMid '{parse['TopicEntityMid']}' not found in KG")
            #    continue
            relation = parse['InferentialChain'][0]
            #try:
            answers = KG[topic_mid][relation]
            #except KeyError:
            #    print(f"KeyError: topic_mid {topic_mid} or relation {relation} not found in KG")
            #    continue
            q[topic_eid] += 1
            answers_rid = np.zeros(len(answers), dtype=int)
            for i, item in enumerate(answers):
                answers_rid[i] = KG.entity_id(item)
                q[answers_rid[i]] += 1 / len(answers)

        for i in range(diameter):
            e += csr_indirect_matrices[i] * q

        # Update relationship heat
        for query in training:
            parse = query['Parse']
            relation = parse['InferentialChain'][0]
            r[KG.relation_id(relation)] += 1

        # Initialize the Heat (H)
        H = {}
        for triple in KG.triples():
            e1, re, e2 = triple
            e1 = KG.entity_id(e1)
            re = KG.relation_id(re)
            e2 = KG.entity_id(e2)
            if e[e1] != 0 and r[re] != 0 and e[e2] != 0:
                H[(e1, re, e2)] = e[e1] * (r[re] + smoothing) * e[e2]
                index_list.append((e1, re, e2))

        # Sort index_list
        h_for_sort = []
        for index in index_list:
            h_for_sort.append(H[index])

        h_for_sort = np.array(h_for_sort)

        sorted_args = np.argsort(h_for_sort)[::-1]
        sorted_index_list = []
        for i in range(len(sorted_args)):
            sorted_index_list.append(index_list[sorted_args[i]])
        index_list = sorted_index_list

        # Construct initial summary
        P = construct_complete(KG, K, index_list)
        run_time = time() - t0

        # Test for initial graph
        if DETAILED_LOGGING:
            logging.info('Initial Test')
        total_F1, total_precision, total_recall = total_query_log_metrics(P, query_log)



        if DETAILED_LOGGING:
            logging.info('\t  Total F1/precision/recall')
            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                total_F1, total_precision, total_recall))

        avg_F1, avg_precision, avg_recall = average_query_log_metrics(P, query_log)
        if DETAILED_LOGGING:
            logging.info('\t  Average F1/precision/recall')
            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                avg_F1, avg_precision, avg_recall))


    # Update phase
    else:
        #print("Adapting for timestamp: ", t)
        t1 = time()

        if H is None:
            H = {}
        if q is None or q.shape[0] != KG.number_of_entities():
            q = np.zeros(KG.number_of_entities())
        if e is None or e.shape[0] != KG.number_of_entities():
            e = np.zeros(KG.number_of_entities())
        if r is None or r.shape[0] != KG.number_of_relationships():
            r = np.zeros(KG.number_of_relationships())

        if (
            csr_indirect_matrices is None
            or 0 not in csr_indirect_matrices
            or csr_indirect_matrices[0].shape[0] != KG.number_of_entities()
        ):
            csr_indirect_matrices = _build_csr_indirect_matrices(KG, diameter, alpha)

        if index_list is None:
            index_list = []

        H_new = {}
        for key in H:
            #print(key)
            H_new[key] = H[key] * gamma**3

        for triple_in_id in H_new:
            if H_new[triple_in_id] > 0 and H_new[triple_in_id] < 1e-5:
                H_new[triple_in_id] = 0

        changed_triples_in_id = set()
        progress_write("Starting query processing")
        # Process all queries in training
        for query in training:
            q_T = np.zeros(KG.number_of_entities())
            parse = query['Parse']
            topic_mid = parse['TopicEntityMid']
            topic_eid = KG.entity_id(parse['TopicEntityMid'])
            #if topic_eid is None:
            #    print(f"Skipping query: TopicEntityMid '{parse['TopicEntityMid']}' not found in KG")
            #    continue
            relation = parse['InferentialChain'][0]
            #try:
            answers = KG[topic_mid][relation]
            #except KeyError:
            #    print(f"KeyError: topic_mid {topic_mid} or relation {relation} not found in KG")
            #    continue
            q_T[topic_eid] += 1
            answers_rid = np.zeros(len(answers), dtype=int)
            for i, item in enumerate(answers):
                answers_rid[i] = KG.entity_id(item)
                q_T[answers_rid[i]] += 1 / len(answers)

            q = q * gamma + q_T
            e_new, changed_entity_list = Heat_Diffuse(e, KG, query, diameter, csr_indirect_matrices)
            r = r * gamma
            r[KG.relation_id(relation)] += 1

            query_changed_triples_in_id = set()

            for changed_entity_id in changed_entity_list:
                potential_triples_vector = KG.csr_matrix_indirect()[changed_entity_id, :]
                rows, cols = potential_triples_vector.nonzero()
                for another_entity_id in cols:
                    if e[another_entity_id] == 0 and e_new[another_entity_id] == 0:
                        continue
                    triple_found = KG.find_triple_by_entities_indirect(changed_entity_id, another_entity_id)
                    if triple_found is not None:
                        if triple_found[1] < KG.number_of_relationships():
                            if triple_found[1] != relation and r[triple_found[1]] == 0:
                                continue
                            query_changed_triples_in_id.add(triple_found)

            e = e_new

            for triple in KG.triples_by_relation_id(KG.relation_id(relation)):
                query_changed_triples_in_id.add(triple)

            changed_triples_in_id.update(query_changed_triples_in_id)
            
        progress_write("Done with query processing")
        for triple_in_id in changed_triples_in_id:
            new_value = e[triple_in_id[0]] * ((r[triple_in_id[1]]) + smoothing) * e[triple_in_id[2]]
            H_new[triple_in_id] = new_value if new_value > 0 else 0

            if triple_in_id not in H:
                H[triple_in_id] = 0

        #incremental
        incremental_sort(index_list, changed_triples_in_id, H, H_new)

        new_index_list = []
        for i in range(len(index_list)):
            if H_new[index_list[i]] != 0:
                new_index_list.append(index_list[i])
            else:
                break
        index_list = new_index_list

        H = {triple_in_id: H_new[triple_in_id] for triple_in_id in index_list}

        #construct
        P = construct_complete(KG, K, index_list)
        run_time = time() - t1
        update_time_list.append(run_time)
        if DETAILED_PRINTING:
            print("Length Index List: ", len(index_list))
            print("Length Changed:", len(changed_triples_in_id))

        if DETAILED_LOGGING:
            logging.info('\t  Adapting for time: {}'.format(t))
            logging.info('\t  Time: {:.2f} seconds'.format(run_time))

        # Evaluate question answering on the testing queries
        total_F1, total_precision, total_recall = total_query_log_metrics(P, query_log)


        if DETAILED_LOGGING:
            logging.info('\t  Total F1/precision/recall')
            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                total_F1, total_precision, total_recall))

        avg_F1, avg_precision, avg_recall = average_query_log_metrics(P, query_log)
        if DETAILED_LOGGING:
            logging.info('\t  Average F1/precision/recall')
            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                avg_F1, avg_precision, avg_recall))

        acc_list.append(avg_F1)

    #logging.info('\t  Ave Time on Each Training Log: {:.2f} seconds'.format(np.mean(update_time_list)))
    #logging.info('\t  Ave Ave F1 on Each Training Log: {:.2f}'.format(np.mean(acc_list)))

    #return acc_list, update_time_list, P
    return P, H, q, e, r, csr_indirect_matrices, index_list, total_F1, run_time






def APEX_(KG, K, training, query_log, query_num_per_test=1, gamma=GAMMA, diameter=1, alpha=0.3, t=0, H=None, q=None, e=None, r=None, csr_indirect_matrices=None, index_list=None):
    print('APEX')
    smoothing = 0.5
    update_time_list = []
    acc_list = []

    if t == 0:
        # Initialization phase
        t0 = time()
        num_queries_total = len(query_log)
        csr_indirect_matrices = {}
        csr_indirect_matrices[0] = sp.eye(KG.number_of_entities())
        for i in range(diameter):
            csr_indirect_matrices[i + 1] = alpha * KG.csr_matrix_indirect_heat() * csr_indirect_matrices[i]

        # Initial state
        index_list = []
        q = np.zeros(KG.number_of_entities())
        e = np.zeros(KG.number_of_entities())
        r = np.zeros(KG.number_of_relationships())

        # Process all queries in training
        for query in training:
            parse = query['Parse']
            topic_mid = parse['TopicEntityMid']
            topic_eid = KG.entity_id(parse['TopicEntityMid'])
            if topic_eid is None:
                print(f"Skipping query: TopicEntityMid '{parse['TopicEntityMid']}' not found in KG")
                continue
            relation = parse['InferentialChain'][0]
            try:
                answers = KG[topic_mid][relation]
            except KeyError:
                print(f"KeyError: topic_mid {topic_mid} or relation {relation} not found in KG")
                continue
            q[topic_eid] += 1
            answers_rid = np.zeros(len(answers), dtype=int)
            for i, item in enumerate(answers):
                answers_rid[i] = KG.entity_id(item)
                q[answers_rid[i]] += 1 / len(answers)

        for i in range(diameter):
            e += csr_indirect_matrices[i] * q

        # Update relationship heat
        for query in training:
            parse = query['Parse']
            relation = parse['InferentialChain'][0]
            r[KG.relation_id(relation)] += 1

        # Initialize the Heat (H)
        H = {}
        for triple in KG.triples():
            e1, re, e2 = triple
            e1 = KG.entity_id(e1)
            re = KG.relation_id(re)
            e2 = KG.entity_id(e2)
            if e[e1] != 0 and r[re] != 0 and e[e2] != 0:
                H[(e1, re, e2)] = e[e1] * (r[re] + smoothing) * e[e2]
                index_list.append((e1, re, e2))

        # Sort index_list
        h_for_sort = []
        for index in index_list:
            h_for_sort.append(H[index])

        h_for_sort = np.array(h_for_sort)

        sorted_args = np.argsort(h_for_sort)[::-1]
        sorted_index_list = []
        for i in range(len(sorted_args)):
            sorted_index_list.append(index_list[sorted_args[i]])
        index_list = sorted_index_list

        # Construct initial summary
        P = construct_complete(KG, K, index_list)
        update_time = time() - t0
        print(update_time, "time taken for initial construction")
        # Test for initial graph
        if DETAILED_LOGGING:
            logging.info('Initial Test')
        total_F1, total_precision, total_recall = total_query_log_metrics(P, query_log)
        print(total_F1)
        if DETAILED_LOGGING:
            logging.info('\t  Total F1/precision/recall')
            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                total_F1, total_precision, total_recall))

        avg_F1, avg_precision, avg_recall = average_query_log_metrics(P, query_log)
        if DETAILED_LOGGING:
            logging.info('\t  Average F1/precision/recall')
            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                avg_F1, avg_precision, avg_recall))

    # Update phase
    else:
        print("Adapting for timestamp: ", t)
        t1 = time()

        # Initialize H, q, e, r, csr_indirect_matrices, and index_list if they are None
        if H is None:
            H = {}
        if q is None:
            q = np.zeros(KG.number_of_entities())
        if e is None:
            e = np.zeros(KG.number_of_entities())
        if r is None:
            r = np.zeros(KG.number_of_relationships())
        if csr_indirect_matrices is None:
            csr_indirect_matrices = {}
            csr_indirect_matrices[0] = sp.eye(KG.number_of_entities())
            for i in range(diameter):
                csr_indirect_matrices[i + 1] = alpha * KG.csr_matrix_indirect_heat() * csr_indirect_matrices[i]
        if index_list is None:
            index_list = []

        H_new = {}
        for key in H:
            H_new[key] = H[key] * gamma**3

        for triple_in_id in H_new:
            if H_new[triple_in_id] > 0 and H_new[triple_in_id] < 1e-5:
                H_new[triple_in_id] = 0

        changed_triples_in_id = set()

        # Process queries incrementally
        for query in training:
            q_T = np.zeros(KG.number_of_entities())
            parse = query['Parse']
            topic_mid = parse['TopicEntityMid']
            topic_eid = KG.entity_id(parse['TopicEntityMid'])
            if topic_eid is None:
                print(f"Skipping query: TopicEntityMid '{parse['TopicEntityMid']}' not found in KG")
                continue
            relation = parse['InferentialChain'][0]
            try:
                answers = KG[topic_mid][relation]
            except KeyError:
                print(f"KeyError: topic_mid {topic_mid} or relation {relation} not found in KG")
                continue
            q_T[topic_eid] += 1
            answers_rid = np.zeros(len(answers), dtype=int)
            for i, item in enumerate(answers):
                answers_rid[i] = KG.entity_id(item)
                q_T[answers_rid[i]] += 1 / len(answers)

            q = q * gamma + q_T
            e_new, changed_entity_list = Heat_Diffuse(e, KG, query, diameter, csr_indirect_matrices)
            r = r * gamma
            r[KG.relation_id(relation)] += 1

            query_changed_triples_in_id = set()

            for changed_entity_id in changed_entity_list:
                potential_triples_vector = KG.csr_matrix_indirect()[changed_entity_id, :]
                rows, cols = potential_triples_vector.nonzero()
                for another_entity_id in cols:
                    if e[another_entity_id] == 0 and e_new[another_entity_id] == 0:
                        continue
                    triple_found = KG.find_triple_by_entities_indirect(changed_entity_id, another_entity_id)
                    if triple_found is not None:
                        if triple_found[1] < KG.number_of_relationships():
                            if triple_found[1] != relation and r[triple_found[1]] == 0:
                                continue
                            query_changed_triples_in_id.add(triple_found)

            e = e_new

            for triple in KG.triples_by_relation_id(KG.relation_id(relation)):
                query_changed_triples_in_id.add(triple)

            changed_triples_in_id.update(query_changed_triples_in_id)

        for triple_in_id in changed_triples_in_id:
            new_value = e[triple_in_id[0]] * ((r[triple_in_id[1]]) + smoothing) * e[triple_in_id[2]]
            H_new[triple_in_id] = new_value if new_value > 0 else 0

            if triple_in_id not in H:
                H[triple_in_id] = 0

        print('incremental')
        incremental_sort(index_list, changed_triples_in_id, H, H_new)

        new_index_list = []
        for i in range(len(index_list)):
            if H_new[index_list[i]] != 0:
                new_index_list.append(index_list[i])
            else:
                break
        index_list = new_index_list

        H = H_new

        print('construct')
        P = construct_complete(KG, K, index_list)
        update_time = time() - t1
        print(update_time, "time taken")
        update_time_list.append(update_time)
        if DETAILED_PRINTING:
            print("Length Index List: ", len(index_list))
            print("Length Changed:", len(changed_triples_in_id))

        if DETAILED_LOGGING:
            logging.info('\t  Adapting for time: {}'.format(t))
            logging.info('\t  Time: {:.2f} seconds'.format(update_time))

        # Evaluate question answering on the testing queries
        total_F1, total_precision, total_recall = total_query_log_metrics(P, query_log)
        print(total_F1)
        if DETAILED_LOGGING:
            logging.info('\t  Total F1/precision/recall')
            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                total_F1, total_precision, total_recall))

        avg_F1, avg_precision, avg_recall = average_query_log_metrics(P, query_log)
        if DETAILED_LOGGING:
            logging.info('\t  Average F1/precision/recall')
            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                avg_F1, avg_precision, avg_recall))

        acc_list.append(avg_F1)

    logging.info('\t  Ave Time on Each Training Log: {:.2f} seconds'.format(np.mean(update_time_list)))
    logging.info('\t  Ave Ave F1 on Each Training Log: {:.2f}'.format(np.mean(acc_list)))

    #return acc_list, update_time_list, P
    return total_F1, update_time, P
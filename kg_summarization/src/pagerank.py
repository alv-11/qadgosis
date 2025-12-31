import numpy as np
import scipy.sparse as sp
from .base import KnowledgeGraph
from time import time
from .metrics import total_query_log_metrics, average_query_log_metrics
from .algorithms import query_vector, random_walk_with_restart
import logging
import pdb
import random

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
        self.entities_.add(topic_mid)

    
    def add_random_triple(self):
        e1 = random.choice(list(self.parent().triples_.keys()))
        r = random.choice(list(self.parent()[e1].keys()))
        e2 = random.choice(list(self.parent()[e1][r]))
        self.add_triple((e1, r, e2))



def PR(KG, K, query_log, query_num_per_test=1, power = 1):
        x = query_vector(KG, query_log)
        M = KG.transition_matrix()
        x = random_walk_with_restart(M, x, power=power)
        sorted_x = np.sort(x)[::-1]
        sorted_args = np.argsort(x)[::-1]

        topic_mids = []
        # turn eid into mid
        for i in range(len(sorted_args)):
            topic_mids.append(KG.id_entity_[sorted_args[i]])

        P = Summary(KG)
        i = 0

        for i in range(len(topic_mids)):
            P.add_entity(topic_mids[i])
            if P.number_of_triples() > K:
                break
            
        return P
            



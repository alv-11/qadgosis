class SummaryGraph(KnowledgeGraph):
    def __init__(self, KG, top_k_nodes):
        """
        Constructs a summary of the knowledge graph based on top-k nodes.

        :param KG: KnowledgeGraph instance (the full KG)
        :param top_k_nodes: List of important nodes (entities) to include in the summary
        """
        super().__init__()
        self.parent_ = KG  
        self.top_k_nodes = set(top_k_nodes)  
        self.build_summary()

    def build_summary(self):
        """
        Builds the summary graph by adding triples that involve top-k nodes.
        """
        for e1 in self.top_k_nodes:
            if e1 not in self.parent().entity_id_:
                print(f"Node {e1} is not in the parent graph.")  # Log nodes not in the parent graph
                continue  # Skip nodes that are not in the original KG

            # Infer forward relations by checking all triples in KG
            if e1 in self.parent().triples_:
                for r, neighbors in self.parent().triples_[e1].items():
                    for e2 in neighbors:
                        if e2 in self.top_k_nodes:  
                            self.add_triple((e1, r, e2))

            # Infer backward relations by checking all triples in KG
            for e2, rel_dict in self.parent().triples_.items():
                for r, neighbors in rel_dict.items():
                    if e1 in neighbors and e2 in self.top_k_nodes: 
                        self.add_triple((e2, r, e1))

    def parent(self):
        return self.parent_

    def add_random_triple(self):
        """ Adds a random triple from the parent KG to the summary """
        e1 = random.choice(list(self.parent().triples_.keys()))
        r = random.choice(list(self.parent().triples_[e1].keys()))
        e2 = random.choice(list(self.parent().triples_[e1][r]))
        self.add_triple((e1, r, e2))

    def validate_top_k_nodes(self):
        """
        Validates if nodes in top_k_nodes exist in the parent graph.
        """
        missing_nodes = [node for node in self.top_k_nodes if node not in self.parent().entity_id_]
        if missing_nodes:
            print(f"Missing nodes in parent graph: {missing_nodes}")
        else:
            print("All top_k_nodes exist in the parent graph.")

class ReasoningTrace:
    def __init__(self, query):
        self.query = query
        self.classification = {}
        self.retrieved_per_subquery = {}
        self.reranked_final = []
        self.generation_prompt = ""
        self.final_answer = ""
        
    def to_dict(self):
        return {
            "query": self.query,
            "classification": self.classification,
            "retrieved_per_subquery": self.retrieved_per_subquery,
            "reranked_final_configs": [r['metadata'] for r in self.reranked_final],
            "generation_prompt": self.generation_prompt,
            "final_answer": self.final_answer
        }

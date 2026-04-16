import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import ContextReRanker
from generation.trace import ReasoningTrace
from generation.generator import FinalGenerator

class ReasoningEngine:
    def __init__(self):
        print("Initializing Reasoning Engine...")
        self.retriever = HybridRetriever()
        self.reranker = ContextReRanker()
        self.generator = FinalGenerator()

    def deduplicate(self, candidates):
        seen = set()
        deduped = []
        for cand in candidates:
            if cand['chunk_id'] not in seen:
                seen.add(cand['chunk_id'])
                deduped.append(cand)
        return deduped

    def commonsense_path(self, trace):
        print("Executing Commonsense Path...")
        query = trace.query
        
        candidates = self.retriever.hybrid_retrieve(query, top_k=20)
        reranked = self.reranker.rerank(query, candidates, top_k=5)
        
        trace.retrieved_per_subquery["main"] = [r['chunk_id'] for r in reranked]
        trace.reranked_final = reranked
        
        return self.generator.generate(trace)

    def adaptive_path(self, trace):
        print("Executing Adaptive Path...")
        sub_questions = trace.classification.get("sub_questions", [])
        
        all_candidates = []
        for sq in sub_questions:
            print(f"Retrieving for sub-question: {sq}")
            cands = self.retriever.hybrid_retrieve(sq, top_k=10)
            ranked = self.reranker.rerank(sq, cands, top_k=3)
            trace.retrieved_per_subquery[sq] = [r['chunk_id'] for r in ranked]
            all_candidates.extend(ranked)
            
        merged = self.deduplicate(all_candidates)
        trace.reranked_final = merged
        
        return self.generator.generate(trace)

    def strategic_path(self, trace):
        print("Executing Strategic Path...")
        sub_questions = trace.classification.get("sub_questions", [])
        
        level1_candidates = self.retriever.hybrid_retrieve(trace.query, top_k=10)
        # Using hierarchical proxy logic:
        # Get category insights, then combine with sub-question retrieve. 
        all_candidates = level1_candidates
        trace.retrieved_per_subquery["level1_main"] = [r['chunk_id'] for r in level1_candidates[:3]]
        
        for sq in sub_questions:
            print(f"Retrieving for sub-category/question: {sq}")
            cands = self.retriever.hybrid_retrieve(sq, top_k=10)
            ranked = self.reranker.rerank(sq, cands, top_k=3)
            trace.retrieved_per_subquery[sq] = [r['chunk_id'] for r in ranked]
            all_candidates.extend(ranked)
            
        merged = self.deduplicate(all_candidates)
        
        # Self-consistency decoding will be triggered in the generator.
        trace.reranked_final = merged
        
        return self.generator.generate(trace)

    def execute(self, trace):
        r_type = trace.classification.get("reasoning_type", "commonsense")
        
        if trace.classification.get("ambiguity", "low") == "high":
            print("Note: High ambiguity detected. Logging assumption.")
            # Trace logger captures this implicitly or generator explicitly mentions it
            
        if r_type == "commonsense":
            return self.commonsense_path(trace)
        elif r_type == "adaptive":
            return self.adaptive_path(trace)
        elif r_type == "strategic":
            return self.strategic_path(trace)
        else:
            return self.commonsense_path(trace)

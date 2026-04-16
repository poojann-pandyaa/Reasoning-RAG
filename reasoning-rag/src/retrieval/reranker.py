import json
from sentence_transformers import CrossEncoder

class ContextReRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading Cross-Encoder model: {model_name}...")
        self.reranker = CrossEncoder(model_name)
        
    def rerank(self, query, candidates, top_k=5):
        """
        candidates: List containing dicts with structure:
            {'chunk_id': int, 'score': float, 'metadata': dict}
        """
        if not candidates:
            return []
            
        # Prepare pairs for cross-encoder
        pairs = [(query, cand['metadata']['chunk_text']) for cand in candidates]
        
        # Predict semantic relevance scores
        scores = self.reranker.predict(pairs)
        
        scored_candidates = []
        for i, cand in enumerate(candidates):
            meta = cand['metadata']
            base_score = float(scores[i])
            
            # Incorporate Stack Exchange preference signals as per requirements
            score_signal = 0.1 * min(meta.get("score", 0) / 100.0, 1.0)
            accepted_signal = 0.15 if meta.get("is_accepted", False) else 0.0
            
            final_score = base_score + score_signal + accepted_signal
            
            scored_candidates.append({
                'chunk_id': cand['chunk_id'],
                'metadata': meta,
                'base_ce_score': base_score,
                'final_score': final_score
            })
            
        # Sort desc by final_score
        ranked = sorted(scored_candidates, key=lambda x: x['final_score'], reverse=True)
        return ranked[:top_k]

if __name__ == "__main__":
    # Test stub
    pass

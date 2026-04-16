import json
from evaluate import load

class Evaluator:
    def __init__(self):
        self.rouge = load("rouge")
        self.bertscore = load("bertscore")
        
    def evaluate_generation(self, predictions, references):
        # ROUGE
        rouge_results = self.rouge.compute(predictions=predictions, references=references)
        
        # BERTScore
        bert_results = self.bertscore.compute(predictions=predictions, references=references, lang="en")
        avg_bert_f1 = sum(bert_results['f1']) / len(bert_results['f1'])
        
        return {
            "rouge": rouge_results,
            "bertscore_f1": avg_bert_f1
        }
        
    def evaluate_retrieval(self, retrieved_chunk_ids, expected_chunk_id, k=5):
        # Recall@k
        top_k = retrieved_chunk_ids[:k]
        return 1.0 if expected_chunk_id in top_k else 0.0

if __name__ == "__main__":
    # Test script for evaluator
    evaluator = Evaluator()
    preds = ["To reverse a list in Python, you can use the reverse() method or slice notation [::-1]."]
    refs = ["You can reverse a list in Python using slicing list[::-1] or the built-in .reverse() method."]
    
    gen_metrics = evaluator.evaluate_generation(preds, refs)
    print("Generation Metrics:", gen_metrics)

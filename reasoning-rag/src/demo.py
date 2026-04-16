import sys
import os
import json

from reasoning.classifier import QueryClassifier
from reasoning.engine import ReasoningEngine
from generation.trace import ReasoningTrace

def run_demo():
    print("Loading Models and Indices... This might take a moment.")
    try:
        classifier = QueryClassifier()
        engine = ReasoningEngine()
    except Exception as e:
        print(f"Failed to initialize system. Are the indices built and the OPENAI_API_KEY set? Error: {e}")
        return

    print("\nType your question below (or 'quit' to exit):")
    while True:
        try:
            query = input("\nQuery: ").strip()
        except EOFError:
            break
            
        if not query:
            continue
            
        if query.lower() in ('quit', 'exit', 'q'):
            break
            
        print("\n" + "="*50)
        print(f"=== Query ===\n{query}\n")
        
        # Phase 1: Classification
        classification = classifier.classify(query)
        intent = classification.get('intent', 'unknown')
        r_type = classification.get('reasoning_type', 'commonsense')
        scope = classification.get('scope', 'unknown')
        sub_questions = classification.get('sub_questions', [])
        
        print(f"=== Classification ===\nIntent: {intent} | Reasoning type: {r_type} | Scope: {scope}\n")
        print("=== Sub-queries issued ===")
        for i, sq in enumerate(sub_questions):
            print(f"{i+1}. {sq}")
        print("\n")
        
        # Phase 2: Engine Execution
        trace = ReasoningTrace(query)
        trace.classification = classification
        
        trace = engine.execute(trace)
        
        # Output Trace
        print("=== Retrieved sources (Top 3) ===")
        for i, cand in enumerate(trace.reranked_final[:3]):
            meta = cand['metadata']
            print(f"[Source {i+1}] Score: {meta.get('score', 0)} | Accepted: {meta.get('is_accepted', False)} | Domain: {meta.get('domain', 'unknown')}")
            # print snippet
            print(f"{meta.get('chunk_text', '')[:150]}...")
            print("-")
            
        print("\n=== Final Answer ===")
        print(trace.final_answer)
        print("="*50 + "\n")

if __name__ == "__main__":
    run_demo()

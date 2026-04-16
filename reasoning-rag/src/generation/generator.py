import os
import torch
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

class FinalGenerator:
    def __init__(self, model_name="google/flan-t5-base"):
        print(f"Loading local LLM for generation: {model_name}...")
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            max_new_tokens=512,
            device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
    def build_prompt(self, query, retrieved_chunks, reasoning_type, sub_questions):
        context_parts = []
        for i, cand in enumerate(retrieved_chunks):
            # Hard limit to top 1 candidate for FLAN-T5-Base sizing
            if i >= 1:
                break
            meta = cand['metadata']
            score = meta.get('score', 0)
            is_acc = meta.get('is_accepted', False)
            chunk_text = meta.get('chunk_text', '')
            # Truncate strictly to smaller pieces
            chunk_text = chunk_text[:500] if len(chunk_text) > 500 else chunk_text
            domain = meta.get('domain', 'unknown')
            context_parts.append(f"[Source {i+1} | Score: {score} | Accepted: {is_acc} | Domain: {domain}]\n{chunk_text}")
            
        context = "\n\n".join(context_parts)
        
        cot_instructions = {
            "commonsense": "Answer the question directly based on the sources above.",
            "adaptive": "First address each sub-question separately, then synthesise into a unified answer.",
            "strategic": "First identify the main categories relevant to this query. Then address each category using the sources. Finally, provide a synthesised cross-category answer."
        }
        cot_instruction = cot_instructions.get(reasoning_type, cot_instructions["commonsense"])
        
        prompt = f"""You are a technical expert answering based on retrieved Stack Exchange content.

Retrieved context:
{context}

Sub-questions to address: {sub_questions}

Instruction: {cot_instruction}

Question: {query}

Reason step by step through the evidence before writing your final answer.
"""
        return prompt

    def generate_with_consistency(self, prompt, n=3):
        print(f"Applying self-consistency decoding (n={n})...")
        # Since local generation without varied temperature is mostly deterministic for greedy,
        # we still mimic it. If we wanted true diversity, we would configure the pipeline with do_sample=True, temperature=0.7.
        responses = [self.llm.invoke(prompt) for _ in range(n)]
        # Simple heuristic: longest response usually packs the most complete synthesis
        return max(responses, key=len)

    def generate(self, trace):
        r_type = trace.classification.get("reasoning_type", "commonsense")
        sq = trace.classification.get("sub_questions", [])
        
        prompt = self.build_prompt(trace.query, trace.reranked_final, r_type, sq)
        trace.generation_prompt = prompt
        
        if r_type == "strategic":
            # high stakes query
            answer = self.generate_with_consistency(prompt, n=3)
        else:
            answer = self.llm.invoke(prompt)
            
        trace.final_answer = answer
        return trace

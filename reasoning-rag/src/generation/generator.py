import os
import torch
from typing import Optional  # Python 3.9 compatible
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from peft import PeftModel
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model_and_tokenizer(
    model_name: str,
    lora_adapter_path: Optional[str],
    device: str,
):
    """
    Load base model (optionally with a LoRA adapter).
    Uses 4-bit quantization on CUDA; plain fp32 on MPS/CPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # MPS / CPU -- bitsandbytes 4-bit not supported; load in float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={"" : device},
            trust_remote_code=True,
        )

    if lora_adapter_path and os.path.isdir(lora_adapter_path):
        print(f"Loading LoRA adapter from {lora_adapter_path} ...")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


class FinalGenerator:
    """
    Drop-in replacement for the old flan-t5-base generator.
    Uses google/gemma-2b-it (causal LM) with optional LoRA adapter.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2b-it",
        lora_adapter_path: Optional[str] = None,
        max_new_tokens: int = 512,
    ):
        print(f"Loading generator model: {model_name} ...")
        device = _get_device()
        print(f"  Device selected: {device}")

        model, tokenizer = _load_model_and_tokenizer(model_name, lora_adapter_path, device)

        hf_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
            device_map="auto" if device == "cuda" else None,
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)
        self.tokenizer = tokenizer

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: list,
        reasoning_type: str,
        sub_questions: list,
    ) -> str:
        """Build a Gemma-IT instruction-following prompt from retrieved evidence."""

        context_parts = []
        for i, cand in enumerate(retrieved_chunks[:3]):
            meta = cand["metadata"]
            score      = meta.get("score", 0)
            is_acc     = meta.get("is_accepted", False)
            chunk_text = meta.get("chunk_text", "")[:800]
            domain     = meta.get("domain", "unknown")
            context_parts.append(
                f"[Source {i+1} | Score: {score} | Accepted: {is_acc} | Domain: {domain}]\n{chunk_text}"
            )

        context = "\n\n".join(context_parts)

        cot_instructions = {
            "commonsense": (
                "Answer the question directly and concisely based on the sources above. "
                "Cite which source you used."
            ),
            "adaptive": (
                "The question has multiple parts. "
                "First address each sub-question separately using the sources, "
                "then synthesise everything into a single unified answer."
            ),
            "strategic": (
                "This is a complex comparative or architectural question. "
                "Step 1 - identify the main categories or dimensions relevant to the query. "
                "Step 2 - discuss each dimension using evidence from the sources. "
                "Step 3 - write a final synthesised recommendation."
            ),
        }
        cot_instruction = cot_instructions.get(reasoning_type, cot_instructions["commonsense"])

        sub_q_block = ""
        if sub_questions and sub_questions != [query]:
            sub_q_block = "Sub-questions to address:\n" + "\n".join(
                f"  {idx+1}. {sq}" for idx, sq in enumerate(sub_questions)
            ) + "\n\n"

        prompt = (
            "<start_of_turn>user\n"
            "You are a senior software engineer answering based strictly on the retrieved Stack Exchange evidence below.\n\n"
            f"Retrieved Evidence:\n{context}\n\n"
            f"{sub_q_block}"
            f"Instruction: {cot_instruction}\n\n"
            f"Question: {query}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        return prompt

    def _score_response(self, response: str) -> float:
        """Lexical diversity score — avoids repetitive answers."""
        tokens = response.split()
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def generate_with_consistency(self, prompt: str, n: int = 3) -> str:
        print(f"Applying self-consistency decoding (n={n}) ...")
        responses = [self.llm.invoke(prompt) for _ in range(n)]
        scored = [(self._score_response(r), r) for r in responses]
        return max(scored, key=lambda x: x[0])[1]

    def generate(self, trace):
        r_type = trace.classification.get("reasoning_type", "commonsense")
        sq     = trace.classification.get("sub_questions", [])

        prompt = self.build_prompt(trace.query, trace.reranked_final, r_type, sq)
        trace.generation_prompt = prompt

        if r_type == "strategic":
            answer = self.generate_with_consistency(prompt, n=3)
        else:
            answer = self.llm.invoke(prompt)

        trace.final_answer = answer
        return trace

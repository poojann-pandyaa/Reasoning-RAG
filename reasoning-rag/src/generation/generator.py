"""
generator.py
------------
FinalGenerator with three backends, selected by priority:
  1. HuggingFace Inference API  -- if HF_TOKEN env var is set (zero RAM in pod)
  2. MLX                        -- Apple Silicon (M1/M2/M3/M4)
  3. PyTorch                    -- last resort (WARNING: OOMs on CPU with Gemma)

Default model: google/gemma-2-2b-it
For fine-tuning, see: src/train_mlx.py
"""

import os
from typing import Optional


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        import mlx_lm    # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# HuggingFace Inference API backend (primary -- no model loaded in pod)
# ---------------------------------------------------------------------------

class _HFAPIGenerator:
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        max_new_tokens: int = 512,
    ):
        from huggingface_hub import InferenceClient
        self._token = os.environ.get("HF_TOKEN")
        if not self._token:
            raise EnvironmentError("[HFAPI] HF_TOKEN env var not set.")
        print(f"[HFAPI] Using HuggingFace Inference API: {model_name}")
        self._client         = InferenceClient(token=self._token)
        self._model          = model_name
        self._max_new_tokens = max_new_tokens

    def invoke(self, prompt: str) -> str:
        response = self._client.text_generation(
            prompt,
            model=self._model,
            max_new_tokens=self._max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
        )
        return response.strip()


# ---------------------------------------------------------------------------
# MLX backend (Apple Silicon fallback)
# ---------------------------------------------------------------------------

class _MLXGenerator:
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        lora_adapter_path: Optional[str] = None,
        max_new_tokens: int = 512,
    ):
        from mlx_lm import load
        print(f"[MLX] Loading model: {model_name} ...")
        if lora_adapter_path and os.path.isdir(lora_adapter_path):
            print(f"[MLX] Loading LoRA adapter from: {lora_adapter_path}")
            self.model, self.tokenizer = load(
                model_name,
                adapter_path=lora_adapter_path,
            )
        else:
            self.model, self.tokenizer = load(model_name)
        self.max_new_tokens = max_new_tokens
        print("[MLX] Model loaded successfully.")

    def invoke(self, prompt: str) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler, make_repetition_penalty

        sampler = make_sampler(temp=0.7, top_p=0.9)
        repetition_penalty = make_repetition_penalty(
            penalty=1.2,
            context_size=20,
        )
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            sampler=sampler,
            logits_processors=[repetition_penalty],
            verbose=False,
        )
        return response.strip()


# ---------------------------------------------------------------------------
# PyTorch fallback (last resort -- will OOM on CPU with Gemma)
# ---------------------------------------------------------------------------

class _TorchGenerator:
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        lora_adapter_path: Optional[str] = None,
        max_new_tokens: int = 512,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"[Torch] Loading model: {model_name} on {device} ...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={"": device},
            trust_remote_code=True,
        )

        if lora_adapter_path and os.path.isdir(lora_adapter_path):
            from peft import PeftModel
            print(f"[Torch] Loading LoRA adapter from: {lora_adapter_path}")
            model = PeftModel.from_pretrained(model, lora_adapter_path)
            model = model.merge_and_unload()

        model.eval()
        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            return_full_text=False,
        )

    def invoke(self, prompt: str) -> str:
        return self._pipe(prompt)[0]["generated_text"].strip()


# ---------------------------------------------------------------------------
# Public FinalGenerator -- auto-selects backend
# Priority: HF API (if HF_TOKEN set) > MLX > PyTorch
# ---------------------------------------------------------------------------

class FinalGenerator:
    """
    FinalGenerator with automatic backend selection.
    - Set HF_TOKEN env var to use HuggingFace Inference API (recommended for k8s).
    - On Apple Silicon without HF_TOKEN, uses MLX natively.
    - Falls back to PyTorch CPU as last resort (not recommended for Gemma -- OOM risk).
    Default model: google/gemma-2-2b-it
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        lora_adapter_path: Optional[str] = None,
        max_new_tokens: int = 512,
    ):
        if os.environ.get("HF_TOKEN"):
            print("[Generator] Backend: HuggingFace Inference API")
            self._backend = _HFAPIGenerator(model_name, max_new_tokens)
        elif _mlx_available():
            print("[Generator] Backend: MLX (Apple Silicon)")
            self._backend = _MLXGenerator(model_name, lora_adapter_path, max_new_tokens)
        else:
            print("[Generator] Backend: PyTorch (WARNING: will OOM on CPU with Gemma)")
            self._backend = _TorchGenerator(model_name, lora_adapter_path, max_new_tokens)

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: list,
        reasoning_type: str,
        sub_questions: list,
    ) -> str:
        context_parts = []
        for i, cand in enumerate(retrieved_chunks[:3]):
            meta       = cand["metadata"]
            score      = meta.get("score", 0)
            is_acc     = meta.get("is_accepted", False)
            chunk_text = meta.get("chunk_text", "")[:1200]
            domain     = meta.get("domain", "unknown")
            context_parts.append(
                f"[Source {i+1} | Score: {score} | Accepted: {is_acc} | Domain: {domain}]\n{chunk_text}"
            )

        context = "\n\n".join(context_parts)

        cot_instructions = {
            "commonsense": (
                "Answer the question thoroughly and helpfully based on the sources above. "
                "Explain the concept, include code examples if the sources contain them, "
                "and cite which source(s) you used. Write at least 3-4 sentences."
            ),
            "adaptive": (
                "The question has multiple parts. "
                "First address each sub-question separately using the sources, "
                "then synthesise everything into a single unified answer. "
                "Be thorough -- include examples and code where relevant."
            ),
            "strategic": (
                "This is a complex comparative or architectural question. "
                "Step 1 - identify the main categories or dimensions relevant to the query. "
                "Step 2 - discuss each dimension in depth using evidence from the sources. "
                "Step 3 - write a final synthesised recommendation with reasoning."
            ),
        }
        cot_instruction = cot_instructions.get(reasoning_type, cot_instructions["commonsense"])

        sub_q_block = ""
        if sub_questions and sub_questions != [query]:
            sub_q_block = "Sub-questions to address:\n" + "\n".join(
                f"  {idx+1}. {sq}" for idx, sq in enumerate(sub_questions)
            ) + "\n\n"

        # Gemma-2 IT chat template
        prompt = (
            "<start_of_turn>user\n"
            "You are a senior software engineer answering Stack Overflow questions. "
            "Use the retrieved evidence below as your primary source, but you may elaborate "
            "on concepts, explain reasoning, and provide structure to make the answer clear.\n\n"
            "RULES:\n"
            "1. Base your answer on the Retrieved Evidence. Do not invent facts that contradict the sources.\n"
            "2. If the sources do not contain enough information to answer, say exactly: "
            "\"The retrieved sources do not contain enough information to answer this question.\" and stop.\n"
            "3. You MAY explain, expand, and structure information from the sources -- "
            "do not copy-paste raw source text verbatim.\n"
            "4. Include code blocks using markdown (```language) if the sources contain code "
            "or if a code example would make the answer significantly clearer.\n\n"
            f"Retrieved Evidence:\n{context}\n\n"
            f"{sub_q_block}"
            f"Instruction: {cot_instruction}\n\n"
            f"Question: {query}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        return prompt

    def _score_response(self, response: str) -> float:
        tokens = response.split()
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def generate_with_consistency(self, prompt: str, n: int = 3) -> str:
        print(f"Applying self-consistency decoding (n={n}) ...")
        responses = [self._backend.invoke(prompt) for _ in range(n)]
        scored    = [(self._score_response(r), r) for r in responses]
        return max(scored, key=lambda x: x[0])[1]

    def generate(self, trace):
        r_type = trace.classification.get("reasoning_type", "commonsense")
        sq     = trace.classification.get("sub_questions", [])

        prompt = self.build_prompt(trace.query, trace.reranked_final, r_type, sq)
        trace.generation_prompt = prompt

        if r_type == "strategic":
            answer = self.generate_with_consistency(prompt, n=3)
        else:
            answer = self._backend.invoke(prompt)

        trace.final_answer = answer
        return trace

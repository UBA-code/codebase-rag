from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generation.citations import Citation, extract_citations
from src.generation.prompt import build_generation_prompt
from src.retrieval.context import AssembledContext


# Container to bundle the model's text along with auditing metrics
@dataclass
class GenerationResult:
    answer: str
    citations: List[Citation]
    valid_citations_count: int
    total_citations_count: int
    prompt_tokens: int
    completion_tokens: int


# Selects the fastest hardware backend available for PyTorch
def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")          # Nvidia GPU
    if torch.backends.mps.is_available():
        return torch.device("mps")           # Apple Silicon GPU (Metal)
    return torch.device("cpu")               # Fallback


class GeneratorEngine:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-0.6B",
        device: Optional[torch.device] = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.model_id = model_id
        self.device = device or resolve_device()

        # CPU does not handle fp16 well; force float32 if running without GPU/MPS
        if self.device.type == "cpu":
            torch_dtype = torch.float32

        # 1. Load tokenizer to encode text into token IDs
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )

        # 2. Load model weights into memory and send them to the target device
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=True,
        ).to(self.device)

        # 3. Disable training-only layers (like Dropout) for consistent output
        self.model.eval()

    def generate(
        self,
        question: str,
        assembled_context: AssembledContext,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GenerationResult:
        # Step 1: Wrap context + question into the ChatML prompt template
        prompt = build_generation_prompt(
            question=question,
            context_text=assembled_context.context_text,
        )

        # Step 2: Convert the prompt text into input ID tensors and move to device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]  # Track input token count

        # Step 3: Set decoding parameters (temp=0.0 means greedy / deterministic)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        # Step 4: Run inference without calculating gradients to save RAM/compute
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Step 5: output_ids contains [prompt + new tokens]; slice off the prompt
        new_tokens = output_ids[0][prompt_len:]

        # Step 6: Convert token IDs back to a readable string (skip control tags)
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Step 7: Regex-check if cited offsets match the chunks given in context
        citations = extract_citations(
            text=answer,
            context_chunks=assembled_context.chunks_used,
        )
        valid_count = sum(1 for c in citations if c.is_valid)

        # Step 8: Return the final response with token counts and citation audit
        return GenerationResult(
            answer=answer,
            citations=citations,
            valid_citations_count=valid_count,
            total_citations_count=len(citations),
            prompt_tokens=prompt_len,
            completion_tokens=len(new_tokens),
        )

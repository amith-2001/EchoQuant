from llama_cpp import Llama
from time import perf_counter

class LLMEngine:
    def __init__(self, model_path, n_gpu_layers=35):
        self.llm = Llama(model_path=model_path,
                         n_gpu_layers=n_gpu_layers,
                         n_ctx=4096,
                         logits_all=False,
                         seed=42)

    def generate(self, prompt, max_tokens=256, temperature=0.7):
        t0 = perf_counter()
        out = self.llm.create_completion(prompt=prompt,
                                         max_tokens=max_tokens,
                                         temperature=temperature)
        t1 = perf_counter()
        text = out["choices"][0]["text"].strip()
        tokens = out["usage"]["completion_tokens"]
        return text, (t1 - t0), tokens / (t1 - t0 + 1e-6)

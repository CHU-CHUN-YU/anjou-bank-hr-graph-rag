# ============================================================
# llm — 本地 HuggingFace 生成模型(LocalHFLLM)
#
# 載入本地 Gemma(4-bit 選配),提供 call_llm_text / call_llm_json。
# transformers/torch 於載入模型時才取用;失敗則回傳 None 由上層 fallback。
# 依賴:config、utils。
# ============================================================

from .config import *
from .utils import *


class LocalHFLLM:
    """
    Colab GPU local LLM wrapper using HuggingFace Transformers.

    Recommended Colab settings:
    - Runtime > Change runtime type > T4 GPU or better
    - Default model: google/gemma-2-2b-it (open source; gated on HF — needs HF token)
    - 4-bit quantization: enabled by default to reduce GPU memory usage

    The wrapper is lazy-loaded: the model is downloaded and loaded only when the
    workflow first needs to generate an answer.
    """
    def __init__(
        self,
        model_name: str = HF_LLM_MODEL_NAME,
        use_4bit: bool = HF_LLM_USE_4BIT,
        max_new_tokens: int = HF_MAX_NEW_TOKENS,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.use_4bit = use_4bit and torch.cuda.is_available()
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading local HF LLM: {model_name}")
        print(f"Device: {self.device}; 4-bit: {self.use_4bit}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True,
            )
        else:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            if not torch.cuda.is_available():
                self.model.to("cpu")

        self.model.eval()

    def _format_messages(self, system_prompt: str, user_prompt: str) -> str:
        system_prompt = system_prompt.strip()
        user_prompt = user_prompt.strip()
        if getattr(self.tokenizer, "chat_template", None):
            # Some chat templates (e.g. Gemma) do NOT support a separate "system" role
            # and raise on it. Try system+user first, then fall back to merging the
            # system prompt into the user turn.
            try:
                return self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                merged = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": merged}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        # Generic fallback for models without a chat_template.
        return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = HF_TEMPERATURE,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        prompt = self._format_messages(system_prompt, user_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # For quantized/device_map=auto models, putting inputs on cuda is usually correct in Colab.
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        do_sample = temperature is not None and temperature > 0
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=0.9 if do_sample else None,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()


_LOCAL_HF_LLM = None

def get_local_hf_llm() -> Optional[LocalHFLLM]:
    global _LOCAL_HF_LLM
    if _LOCAL_HF_LLM is None:
        try:
            _LOCAL_HF_LLM = LocalHFLLM()
        except Exception as e:
            print("Local HF LLM loading failed. Fallback to template answer.", repr(e))
            _LOCAL_HF_LLM = None
    return _LOCAL_HF_LLM


def call_llm_text(system_prompt: str, user_prompt: str, temperature: float = 0.1, max_new_tokens: int = HF_MAX_NEW_TOKENS) -> Optional[str]:
    llm = get_local_hf_llm()
    if llm is None:
        return None
    try:
        return llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    except Exception as e:
        print("Local HF generation failed. Fallback to template answer.", repr(e))
        return None


def extract_json_from_text(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def call_llm_json(system_prompt: str, user_prompt: str, default: dict) -> dict:
    # Local small LLM JSON can be unstable, so this is optional and defaults to heuristic result.
    text = call_llm_text(system_prompt + "\n請只輸出 valid JSON，不要輸出 markdown。", user_prompt, temperature=0.0, max_new_tokens=384)
    parsed = extract_json_from_text(text)
    return parsed if isinstance(parsed, dict) else default

# -----------------------------
# 8. LangGraph Workflow
# -----------------------------

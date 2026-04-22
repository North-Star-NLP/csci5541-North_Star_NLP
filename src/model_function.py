import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

def load_reader_model(
    model_name: str = "HuggingFaceH4/zephyr-7b-beta",
    local_path: str = "models/zephyr-7b-beta",
    do_sample: bool = True,
    temperature: float = 0.2,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 500,
):
    """
    Load a reader LLM, saving locally on first download.

    Args:
        model_name: Model identifier from HuggingFace Hub
        local_path: Local path to save/load the model
        do_sample: Whether to use sampling for generation
        temperature: Sampling temperature (None for greedy decoding)
        repetition_penalty: Penalty for repeated tokens
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        A transformers text-generation pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    if os.path.exists(local_path):
        print(f"Loading model from local '{local_path}' on {device}...")
        model = AutoModelForCausalLM.from_pretrained(local_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(local_path)
    else:
        print(f"Downloading model '{model_name}' on {device}...")
        if device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
        print(f"Model saved to '{local_path}'")

    return pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=do_sample,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
    )


def load_judge_model(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    local_path: str = "models/qwen2.5-3b-judge",
    do_sample: bool = False,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
):
    """
    Load a judge LLM, saving locally on first download.

    Args:
        model_name: Model identifier from HuggingFace Hub
        local_path: Local path to save/load the model
        do_sample: Whether to use sampling for generation
        temperature: Sampling temperature (0.0 for greedy decoding)
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        A transformers text-generation pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    if os.path.exists(local_path):
        print(f"Loading judge model from local '{local_path}' on {device}...")
        model = AutoModelForCausalLM.from_pretrained(local_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(local_path)
    else:
        print(f"Downloading judge model '{model_name}' on {device}...")
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
        print(f"Judge model saved to '{local_path}'")

    return pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=do_sample,
        temperature=temperature,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
    )
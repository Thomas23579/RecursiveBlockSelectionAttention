"""
Gemma 4 E2B – minimal inference demo
Import once, then call run() as many times as you like:
    import testModel
    testModel.run("your prompt here")
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from logattention import patch_global_attention

# MODEL_ID = "google/gemma-4-31B-it"  # instruction-tuned variant, requires H200 GPU or similar
# MODEL_ID = "google/gemma-4-e4b-it"  # instruction-tuned variant
MODEL_ID = "google/gemma-4-e2b-it"  # instruction-tuned variant

# Replace the attention mechanism at runtime? (not necessary if `unsafe_replace_gemma4_model.py` has been executed)
_monkey_patch = True

# module-level cache so weights load only once per process
_tokenizer = None
_model = None

def load_model(model_id=MODEL_ID, patch=True):
    """Load tokenizer + model once and cache at module level."""
    global _tokenizer, _model
    if _model is None:
        print(f"Loading {model_id} …")
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,   # native precision, halves memory vs float32
            device_map="cuda",
        )
        if patch:
            patch_global_attention(_model)
    return _tokenizer, _model


def run(prompt, system="You are a helpful assistant.",
        max_new_tokens=512, block_size=None, patch=True, stream=True):
    """Generate a completion for a single prompt and return the new text."""
    tokenizer, model = load_model(patch=patch)
    if block_size is not None:
        model.config.text_config.block_size = block_size

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=False,
    ).to(model.device)

    streamer = (TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                if stream else None)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            do_sample=True,
            streamer=streamer,
        )

    # slice off the prompt, return only the generated tokens as text
    new_tokens = output_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    prompt = r"Write down the equations for the Kalman Filter for a discrete-time system of the form $x_{k+1} = A x_k + B u_k + G \nu_k$, $y_k = C x_k + \mu_k$. Write it using Markdown."
    run(prompt, max_new_tokens = 10, block_size=512, patch=_monkey_patch)
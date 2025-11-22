# %%
%load_ext autoreload
%autoreload 2
# %%
import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import activation_additions as aa
from typing import List, Dict, Union, Callable, Tuple
from functools import partial, lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_additions.compat import ActivationAddition, get_x_vector, print_n_comparisons, get_n_comparisons, pretty_print_completions
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
# %%
_ = torch.set_grad_enabled(False)
# %%
model_path: str = "../models/gpt-oss-20b"
device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.tie_weights() # in case checkpoint doesn't contain duplicate keys for tied weights

# {0: '20G', 1: '20G'}
model = load_checkpoint_and_dispatch(model, model_path, device_map={'': device}, dtype=torch.float16, no_split_module_classes=["GPT2Block"])
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.tokenizer = tokenizer
# %%
sampling_kwargs: Dict[str, Union[float, int]] = {
    "temperature": 1.0,
    "top_p": 0.3,
    "freq_penalty": 1.0,
    "num_comparisons": 3,
    "tokens_to_generate": 50,
    "seed": 0,  # For reproducibility
}
get_x_vector_preset: Callable = partial(
    get_x_vector,
    pad_method="tokens_right",
    model=model,
    custom_pad_id=int(model.tokenizer.encode(" ")[0]),
)

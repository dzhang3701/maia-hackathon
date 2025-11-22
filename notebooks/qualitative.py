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
from transformers import LlamaForCausalLM, LlamaTokenizer
from activation_additions.compat import ActivationAddition, get_x_vector, print_n_comparisons, get_n_comparisons, pretty_print_completions
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
# %%
_ = torch.set_grad_enabled(False)
# %%
model_path: str = "../models/llama-13B"
device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

with init_empty_weights():
    model = LlamaForCausalLM.from_pretrained(model_path)#, load_in_8bit=True, device_map={'': device})
    model.tie_weights() # in case checkpoint doesn't contain duplicate keys for tied weights

# {0: '20G', 1: '20G'}
model = load_checkpoint_and_dispatch(model, model_path, device_map={'': device}, dtype=torch.float16, no_split_module_classes=["LlamaDecoderLayer"])
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model.tokenizer = tokenizer
# %%

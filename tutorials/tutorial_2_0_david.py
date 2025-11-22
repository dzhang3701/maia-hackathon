# %%
try:
    import google.colab  # type: ignore
    from google.colab import output

    COLAB = True
    %pip install sae-lens transformer-lens sae-dashboard
except:
    COLAB = False
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# Standard imports
import os
import torch
from tqdm.auto import tqdm
import plotly.express as px
import pandas as pd

# Imports for displaying vis in Colab / notebook

torch.set_grad_enabled(False)

# For the most part I'll try to import functions and classes near where they are used
# to make it clear where they come from.

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
# %%
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

# TODO: Make this nicer.
df = pd.DataFrame.from_records(
    {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
).T
df.drop(
    columns=[
        "expected_var_explained",
        "expected_l0",
        "config_overrides",
        "conversion_func",
    ],
    inplace=True,
)
df  # Each row is a "release" which has multiple SAEs which may have different configs / match different hook points in a model.
# %%
# show the contents of the saes_map column for a specific row
print("SAEs in the GTP2 Small Resid Pre release")
for k, v in df.loc[df.release == "gpt2-small-res-jb", "saes_map"].values[0].items():
    print(f"SAE id: {k} for hook point: {v}")

print("-" * 50)
print("SAEs in the feature splitting release")
for k, v in (
    df.loc[df.release == "gpt2-small-res-jb-feature-splitting", "saes_map"]
    .values[0]
    .items()
):
    print(f"SAE id: {k} for hook point: {v}")

print("-" * 50)
print("SAEs in the Gemma base model release")
for k, v in df.loc[df.release == "gemma-2b-res-jb", "saes_map"].values[0].items():
    print(f"SAE id: {k} for hook point: {v}")
# %%
# from transformer_lens import HookedTransformer
from sae_lens import SAE, HookedSAETransformer

model = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",  # <- Release name
    sae_id="blocks.7.hook_resid_pre",  # <- SAE id (not always a hook point!)
    device=device,
)
# %%
print(sae.cfg.__dict__)
# %%
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate

dataset = load_dataset(
    path="NeelNanda/pile-10k",
    split="train",
    streaming=False,
)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=sae.cfg.metadata.context_size,
    add_bos_token=sae.cfg.metadata.prepend_bos,
)
# %%
from IPython.display import IFrame

# get a random feature from the SAE
feature_idx = torch.randint(0, sae.cfg.d_sae, (1,)).item()

html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"


def get_dashboard_html(sae_release="gpt2-small", sae_id="7-res-jb", feature_idx=0):
    return html_template.format(sae_release, sae_id, feature_idx)


html = get_dashboard_html(
    sae_release="gpt2-small", sae_id="7-res-jb", feature_idx=feature_idx
)
IFrame(html, width=1200, height=600)
# %%
# instantiate an object to hold activations from a dataset
from sae_lens import ActivationsStore

# a convenient way to instantiate an activation store is to use the from_sae method
activation_store = ActivationsStore.from_sae(
    model=model,
    dataset=sae.cfg.metadata.dataset_path,
    sae=sae,
    streaming=True,
    # fairly conservative parameters here so can use same for larger
    # models without running out of memory.
    store_batch_size_prompts=8,
    train_batch_size_tokens=4096,
    n_batches_in_buffer=32,
    device=device,
)
# %%

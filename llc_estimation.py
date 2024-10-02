# %%
import os
import pickle
from functools import partial

import torch
from tqdm import tqdm

from dgp import get_dataloader
from utils.default_device import get_default_device
from utils.llc import calculate_llc_for_file, evaluate_fn
from utils.loading import Conf, load_model_from_hf

# %%

device = "cpu"  # get_default_device()

hf_repo_name = "cybershiptrooper/ConceptPerlocation_ckpts_98k"
dump_dir = f"results/scratch/{hf_repo_name}/llc"
model_dir = f"results/scratch/{hf_repo_name}"


os.makedirs(dump_dir, exist_ok=True)

# %%

model = load_model_from_hf(0, hf_repo_name, epoch=0)
config = Conf(**model["config"])

config.device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = get_dataloader(
    n_relative_properties=config.data.n_relative_properties,
    n_descriptive_properties=config.data.n_descriptive_properties,
    n_descriptive_values=config.data.n_descriptive_values,
    num_of_classes_to_divide_over=config.data.num_of_classes_to_divide_over,
    prior_param=config.data.prior_param,
    props_prior_type=config.data.props_prior_type,
    n_entities=config.data.n_entities,
    instr_ratio=config.data.instr_ratio,
    max_sample_length=config.data.max_sample_length,
    num_iters=5e5 * config.data.batch_size,
    batch_size=config.data.batch_size,
    num_workers=config.data.num_workers,
    seed=config.seed,
    evaluation=True,
)

# %%
pad_token_id = dataloader.dataset.pad_token_id
evaluator = evaluate_fn  # , pad_token_id=pad_token_id, config=config)

# %%
llc_outputs = []
iters = (
    list(range(0, 200, 10))
    + list(range(200, 1200, 20))
    + list(range(1200, 5_001, 50))
    + list(range(5_100, 10_001, 100))
    + list(range(10_500, 25_501, 500))
    + list(range(26_000, 100_501, 1000))
)
for iter in tqdm(iters):
    print(f"Calculating LLC for iteration {iter}")
    llc_output = calculate_llc_for_file(
        iter,
        dataloader,
        model_dir=hf_repo_name,
        config=config,
        evaluate_fn=evaluator,
        model_loader=load_model_from_hf,
        num_chains=5,
        num_draws=200,
    )
    llc_outputs.append(llc_output)
    with open(f"{dump_dir}/llc_output_it_{iter}.pkl", "wb") as f:
        pickle.dump(llc_output, f)

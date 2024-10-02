# %%
import os
import pickle
import time
from functools import partial

import torch

from dgp import get_dataloader
from utils.default_device import get_default_device
from utils.llc import calculate_llc_for_file, evaluate_fn
from utils.loading import Conf, load_model_from_hf
import torch.nn.functional as F
from utils import move_to_device
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset


# %%

# device = "cpu"
device = get_default_device()

hf_repo_name = "cybershiptrooper/ConceptPerlocation_ckpts_98k"
dump_dir = f"results/scratch/{hf_repo_name}/llc"
model_dir = f"results/scratch/{hf_repo_name}"


os.makedirs(dump_dir, exist_ok=True)

# %%

model = load_model_from_hf(0, hf_repo_name, epoch=0)
config = Conf(**model["config"])

config.device = device

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
)
pad_token_id = dataloader.dataset.pad_token_id

all_inputs = []
all_labels = []
all_masks = []
for i, batch in tqdm(enumerate(dataloader)):
    if i > 100:
        break
    sequences, symb_sequences, seq_lengths, seq_logprobs, _ = batch
    inputs = sequences[:, :-1]
    labels = sequences[:, 1:].clone()
    # procesed_batch = [inputs, labels]
    all_inputs.append(inputs)
    all_labels.append(labels)

    mask = (labels != pad_token_id).float()
    all_masks.append(mask)

# make a dataset with all the inputs and labels
print("Making a new dataloader with all the data")
all_inputs = torch.cat(all_inputs, dim=0)
all_labels = torch.cat(all_labels, dim=0)
all_masks = torch.cat(all_masks, dim=0)
print(all_inputs.shape, all_labels.shape)
print(all_inputs.device, all_labels.device)
new_dataloader = DataLoader(
    TensorDataset(all_inputs, all_labels, all_masks),
    batch_size=config.data.batch_size,
    num_workers=config.data.num_workers,
    shuffle=True,
)
print("Done making the new dataloader")


def evaluate_fn_new(model, data):
    inputs, labels, mask = data
    logits = model(inputs)  # (B, L-1, V)
    loss = F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        reduction="none",
    )  # (B*L-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss, {}


# %%
evaluator = evaluate_fn_new

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
    time_now = time.time()
    llc_output = calculate_llc_for_file(
        iter,
        new_dataloader,
        model_dir=hf_repo_name,
        config=config,
        evaluate_fn=evaluator,
        model_loader=load_model_from_hf,
        num_chains=5,
        num_draws=200,
        vocab_size=dataloader.dataset.PCSG.vocab_size,
    )
    time_taken = time.time() - time_now
    # print formatted time
    print(f"Time taken: {time_taken//60:.0f}m {time_taken%60:.0f}s")
    llc_outputs.append(llc_output)
    with open(f"{dump_dir}/llc_output_it_{iter}.pkl", "wb") as f:
        pickle.dump(llc_output, f)

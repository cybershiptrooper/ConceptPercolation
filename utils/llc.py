import typing
from typing import Type

import numpy as np
import torch
import torch.nn.functional as F
from devinterp.optim import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary

from model import GPT
from utils import move_to_device

from .loading import load_model_for_iteration


def evaluate_fn(model, data, pad_token_id, config):
    sequences, symb_sequences, seq_lengths, seq_logprobs, _ = data
    B = sequences.size(0)
    inputs, labels = move_to_device([sequences[:, :-1], sequences[:, 1:]], config.device)
    labels = labels.clone()
    labels[labels == pad_token_id] = -100  # Mask padding
    logits = model(inputs)  # (B, L-1, V)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )  # (B*L-1)
    loss = loss.reshape(B, -1).mean()
    return loss, {}


def estimate_llc_for_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    evaluate: typing.Callable,
    epsilon: float,
    beta: float,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    localization: float = 100.0,
    num_chains: int = 5,
    num_draws: int = 300,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    device: torch.device | str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    online: bool = True,
    verbose: bool = False,
):
    sweep_stats = estimate_learning_coeff_with_summary(
        model,
        loader=loader,
        evaluate=evaluate,
        sampling_method=sampling_method,
        optimizer_kwargs=dict(lr=epsilon, localization=localization, nbeta=beta),
        num_chains=num_chains,  # How many independent chains to run
        num_draws=num_draws,  # How many samples to draw per chain
        num_burnin_steps=num_burnin_steps,  # How many samples to discard at the beginning of each chain
        num_steps_bw_draws=num_steps_bw_draws,  # How many steps to take between each sample
        device=device,
        online=online,
        verbose=verbose,
    )

    sweep_stats["llc/trace"] = np.array(sweep_stats["llc/trace"])
    return sweep_stats


def calculate_llc_for_file(
    iteration,
    dataloader,
    model_dir,
    config,
    evaluate_fn: typing.Callable,
    model_loader: typing.Callable = load_model_for_iteration,
    optimizer_kwargs: dict = dict(lr=1e-3, localization=200.0, nbeta=30),
    num_chains: int = 5,
    num_draws: int = 100,
):
    device = config.device
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.empty_cache()
    model_info = model_loader(iteration, model_dir, epoch=0)
    model = GPT(config.model, dataloader.dataset.PCSG.vocab_size)
    model.load_state_dict(model_info["net"])
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.empty_cache()
    return estimate_learning_coeff_with_summary(
        model,
        loader=dataloader,
        evaluate=evaluate_fn,
        sampling_method=SGLD,
        optimizer_kwargs=optimizer_kwargs,
        num_chains=num_chains,
        num_draws=num_draws,
        num_burnin_steps=0,
        num_steps_bw_draws=1,
        device=config.device,
        online=True,
        verbose=True,
    )

import typing

import torch.nn.functional as F
from devinterp.optim import SGLD
from devinterp.slt.llc import LLCEstimator
from devinterp.slt.sampler import sample
from model import GPT

from .loading import load_model_for_iteration


def evaluate_fn(model, data):
    inputs, labels, mask = data
    logits = model(inputs)
    loss = F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        reduction="none",
    )
    loss = (loss * mask).mean()
    return loss, {}


def evaluate_fn_preloaded_no_mask(model, data):
    inputs, labels = data
    B = inputs.size(0)
    logits = model(inputs)  # (B, L-1, V)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )  # (B*L-1)
    loss = loss.reshape(B, -1).mean()
    return loss, {}

def calculate_llc_for_file(
    iteration,
    dataloader,
    model_dir,
    config,
    model_loader: typing.Callable = load_model_for_iteration,
    optimizer_kwargs: dict = dict(lr=1e-3, localization=200.0, nbeta=30),
    num_chains: int = 4,
    num_draws: int = 100,
    vocab_size: int = None,
    device="cpu",
    evaluate_fn: typing.Callable = evaluate_fn,
    **kwargs,
):

    model_info = model_loader(iteration, model_dir, epoch=0)
    if vocab_size is None:
        try:
            vocab_size = dataloader.dataset.PCSG.vocab_size
        except AttributeError:
            raise ValueError(
                "vocab_size must be provided if dataloader does not have PCSG attribute"
            )
    model = GPT(config.model, vocab_size)
    model.load_state_dict(model_info["net"])
    result = sample(
        model,
        dataloader,
        evaluate=evaluate_fn,
        sampling_method=SGLD,
        callbacks=[
            LLCEstimator(
                num_chains,
                num_draws,
                nbeta=optimizer_kwargs["nbeta"],
                device=device,
                init_loss=0.0,
            )
        ],
        optimizer_kwargs=optimizer_kwargs,
        num_draws=num_draws,
        num_chains=num_chains,
        device=device,
        **kwargs,
    )
    # TODO write result to a file, or log to W&B, etc.
    return result

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

import wandb
import io


def plot_trace(
    llcs,
    eps,
    gammas,
    version: Literal["normal", "mean_std", "min_max"] = "normal",
    figsize=(20, 20),
    log_to_wandb: bool = True,
):
    fig, ax = plt.subplots(len(eps), len(gammas), figsize=figsize)
    std, mean, mins, maxs = {}, {}, {}, {}

    for i, e in enumerate(eps):
        for j, g in enumerate(gammas):
            llc_output = llcs[(e, g)]
            traces = llc_output["loss/trace"]
            all_mean = np.mean(traces)
            ax[i, j].hlines(all_mean, 0, len(traces[0]), color="red", linestyle="--", linewidth=2)
            if version == "normal":
                cmap = plt.get_cmap("viridis")
                for color, chain in enumerate(traces):
                    ax[i, j].plot(chain, alpha=0.7, color=cmap(color / len(traces)))

            elif version == "mean_std":
                # Calculate mean and standard deviation
                trace_mean = np.mean(traces, axis=0)
                trace_std = np.std(traces, axis=0)
                ax[i, j].plot(trace_mean, color="blue")
                ax[i, j].fill_between(
                    np.arange(len(trace_mean)),
                    trace_mean - trace_std,
                    trace_mean + trace_std,
                    color="blue",
                    alpha=0.3,
                )

            elif version == "min_max":
                # Calculate min and max for traces
                trace_mean = np.mean(traces, axis=0)
                trace_min = np.min(traces, axis=0)
                trace_max = np.max(traces, axis=0)
                ax[i, j].plot(trace_mean, color="blue")
                ax[i, j].fill_between(
                    np.arange(len(trace_mean)),
                    trace_min,
                    trace_max,
                    color="blue",
                    alpha=0.3,
                )

            ax[i, j].set_title(f"e: {e}, g: {g}")

    # Set font size for all subplots
    for a in ax.flatten():
        a.tick_params(axis="both", which="major", labelsize=16)
        a.set_title(a.get_title(), fontsize=16)

    plt.tight_layout()
    
    # Log the figure to wandb and return the image
    if log_to_wandb:
        wandb.log(
            {
                f"trace_eps_{'-'.join(map(str, eps))}_gamma_{'-'.join(map(str, gammas))}_{version}": fig
            }
        )
    return fig

def plot_variance_vs_mean(llcs, eps, gammas, figsize=(10, 10), log_to_wandb: bool = True):
    fig, ax = plt.subplots(len(eps), len(gammas), figsize=figsize)
    for i, e in enumerate(eps):
        for j, g in enumerate(gammas):
            llc_output = llcs[(e, g)]
            traces = llc_output["loss/trace"]
            all_mean = np.mean(traces, axis=0)
            all_std = np.std(traces, axis=0)

            # plot 1/snr curve
            snr = all_std / all_mean
            ax[i, j].scatter(all_mean, snr, color="red", alpha=0.5, marker=".")
            ax[i, j].set_title(f"e: {e}, g: {g}")
            ax[i, j].set_xlabel("Mean")

    plt.tight_layout()
    
    # Log the figure to wandb and return the image
    if log_to_wandb:
        wandb.log(
            {
                f"variance_vs_mean_eps_{'-'.join(map(str, eps))}_gamma_{'-'.join(map(str, gammas))}": fig
            }
        )
    return fig

def plot_normalised_stds(llcs, eps, gammas, llc=True, log_to_wandb: bool = True):
    image_grid = np.zeros((len(eps), len(gammas)))
    for i, e in enumerate(eps):
        for j, g in enumerate(gammas):
            llc_output = llcs[(e, g)]
            traces = llc_output["loss/trace"]
            if llc:
                trace_mean = llc_output["llc/means"].mean()
                trace_std = llc_output["llc/stds"].mean()
                nvar = trace_std / trace_mean
            else:
                trace_mean = np.mean(traces, axis=0)
                trace_std = np.std(traces, axis=0)
                nvar = (trace_std / (trace_mean + 1e-6)).mean()
            image_grid[i, j] = nvar
    
    fig, ax = plt.subplots()
    im = ax.imshow(image_grid, cmap="viridis")
    plt.colorbar(im)
    plt.xticks(np.arange(len(gammas)), gammas)
    plt.yticks(np.arange(len(eps)), eps)
    plt.xlabel("Gamma")
    plt.ylabel("Epsilon")
    
    # Log the figure to wandb and return the image
    if log_to_wandb:
        wandb.log(
            {
                f"normalised_stds_eps_{'-'.join(map(str, eps))}_gamma_{'-'.join(map(str, gammas))}": fig
            }
        )
    return fig
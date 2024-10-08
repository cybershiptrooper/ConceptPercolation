import os
import pickle

import torch.multiprocessing as mp
import wandb

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    pass

import tempfile

from dgp import get_dataloader_and_vocab_size, make_preloaded_dataset
from torch.utils.data import DataLoader
from utils.llc import calculate_llc_for_file
from utils.loading import Conf, load_model_from_hf
from utils.plotting import plot_normalised_stds, plot_trace, plot_variance_vs_mean


def process_iteration(
    params, dataloader, hf_repo_name, config, device, num_chains=8, **kwargs
):
    iteration, epsilon, gamma = params
    print(f"Processing: iteration={iteration}, epsilon={epsilon}, gamma={gamma}")

    llc_output = calculate_llc_for_file(
        iteration=iteration,
        dataloader=dataloader,
        model_dir=hf_repo_name,
        config=config,
        model_loader=load_model_from_hf,
        num_chains=num_chains,
        num_draws=200,
        device=device,
        epsilon=epsilon,
        gamma=gamma,
        **kwargs,
    )

    return (epsilon, gamma), llc_output


def log_llc_to_wandb(iteration, epsilon, gamma, llc_output):
    print("logging to wandb", f"llc_output_it{iteration}_e{epsilon}_g{gamma}")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        pickle.dump(llc_output, temp_file)
        temp_file_full_path = temp_file.name
        temp_file_dir = os.path.dirname(temp_file.name)
        wandb.save(glob_str=temp_file_full_path, base_path=temp_file_dir)

    print("logged to wandb")


def main_tpu(tpu_idx, params, hf_repo_name, config, dataset, vocab_size, results_dict):
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=True,
    )
    key, results = process_iteration(
        params,
        dataloader,
        hf_repo_name,
        config,
        device=xm.xla_device(),
        vocab_size=vocab_size,
    )
    results_dict[key] = results


def main_cpu(params_list, hf_repo_name, config):
    dataloader, vocab_size = get_dataloader_and_vocab_size(
        config=config, preloaded_batches=True, eval=True
    )
    results = {}
    for params in params_list:
        key, llc_output = process_iteration(
            params,
            dataloader,
            hf_repo_name,
            config,
            device="cpu",
            num_chains=4,
            vocab_size=vocab_size,
            cores=4,
        )
        results[key] = llc_output
        log_llc_to_wandb(params[0], params[1], params[2], llc_output)
    return results


def main(iteration, epsilons, gammas, hf_repo_name, config, run_on, use_wandb=True):
    if use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project="concept-percolation-llc-sweep",
            config=config,
            name=f"{hf_repo_name}_icl1_style_sweep_{iteration}",
        )

    params_list = [(iteration, e, g) for e in epsilons for g in gammas]

    if run_on == "tpu":
        print("Running on TPU")
        default_loader, vocab_size = get_dataloader_and_vocab_size(
            config=config, preloaded_batches=False
        )
        preloaded_dataset = make_preloaded_dataset(default_loader, num_batches=250)
        manager = mp.Manager()
        results_dict = manager.dict()
        for params in params_list:
            xmp.spawn(
                main_tpu,
                args=(
                    params,
                    hf_repo_name,
                    config,
                    preloaded_dataset,
                    vocab_size,
                    results_dict,
                ),
                join=True,
            )
            key = (params[1], params[2])
            log_llc_to_wandb(params[0], params[1], params[2], results_dict[key])
        results = dict(results_dict)
    elif run_on == "cpu":
        print("Running on CPU/Multiprocessing")
        print("Make sure USE_TPU_BACKEND is set to 0 in your .env")
        results = main_cpu(params_list, hf_repo_name, config)
    else:
        raise NotImplementedError

    # Generate and save plots
    plot_trace(results, epsilons, gammas)
    plot_variance_vs_mean(results, epsilons, gammas)
    plot_normalised_stds(results, epsilons, gammas, False)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    run_on = "tpu"  # or 'cpu'

    hf_repo_name = "cybershiptrooper/ConceptPerlocation_ckpts_98k"
    model_dir = f"results/scratch/{hf_repo_name}"

    model = load_model_from_hf(0, hf_repo_name, epoch=0)
    config = Conf(**model["config"])
    del model

    iteration = 20_000
    epsilons = [5e-4, 1e-3, 5e-3]
    gammas = [100.0, 200.0, 400.0]

    # Call the main function with the appropriate environment
    main(iteration, epsilons, gammas, hf_repo_name, config, run_on, use_wandb=True)

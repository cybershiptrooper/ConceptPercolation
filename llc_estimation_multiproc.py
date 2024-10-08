import wandb
import io
import pickle
import dotenv
import os

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    pass

from dgp import get_dataloader_and_vocab_size, make_preloaded_dataset
from utils.llc import calculate_llc_for_file
from utils.loading import Conf, load_model_from_hf
from torch.utils.data import DataLoader


def process_iteration(it_no, dataloader, hf_repo_name, config, device, num_chains=8, **kwargs):
    llc_output = calculate_llc_for_file(
        it_no,
        dataloader,
        model_dir=hf_repo_name,
        config=config,
        model_loader=load_model_from_hf,
        num_chains=num_chains,  # should be same as number of TPU cores
        num_draws=200,
        device=device,
        **kwargs,
    )
    
    # Log LLC output to Wandb
    log_llc_to_wandb(it_no, llc_output)
    
    return llc_output


def log_llc_to_wandb(it_no, llc_output):
    # Serialize LLC output using pickle
    llc_pickle = pickle.dumps(llc_output)
    
    # Create a file-like object in memory
    llc_file = io.BytesIO(llc_pickle)
    
    # Log the file to Wandb
    wandb.log({f"llc_output_{it_no}": wandb.File(llc_file, name=f"llc_output_{it_no}.pkl")})


def main_tpu(tpu_idx, it_no, hf_repo_name, config, dataset, vocab_size):
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=True,
    )
    process_iteration(
        it_no, dataloader, hf_repo_name, config, device=xm.xla_device(), vocab_size=vocab_size
    )


def main_cpu(iters, hf_repo_name, config):
    dataloader, vocab_size = get_dataloader_and_vocab_size(
        config=config, preloaded_batches=True, eval=True
    )
    for it_no in iters:
        llc_output = process_iteration(
            it_no,
            dataloader,
            hf_repo_name,
            config,
            device="cpu",
            num_chains=4,
            vocab_size=vocab_size,
            cores=4,
        )


# Main function that determines the execution environment
def main(iters, hf_repo_name, config, run_on, use_wandb=True):
    dotenv.load_dotenv("~/timaeus/.env")
    if use_wandb:
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        wandb.init(project="concept-percolation-llc", config=config, name=hf_repo_name)
    
    if run_on == "tpu":
        print("Running on TPU")
        default_loader, vocab_size = get_dataloader_and_vocab_size(
            config=config, preloaded_batches=False
        )
        preloaded_dataset = make_preloaded_dataset(default_loader)
        for it_no in iters:
            xmp.spawn(main_tpu, args=(it_no, hf_repo_name, config, preloaded_dataset, vocab_size))
    elif run_on == "cpu":
        print("Running on CPU/Multiprocessing")
        print("Make sure USE_TPU_BACKEND is set to 0 in your .env")
        main_cpu(iters, hf_repo_name, config)
    else:
        raise NotImplementedError
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    run_on = "tpu"  # or 'cpu'. note that this is NOT the device the process ultimately runs on!

    hf_repo_name = "cybershiptrooper/ConceptPerlocation_ckpts_98k"
    model_dir = f"results/scratch/{hf_repo_name}"

    model = load_model_from_hf(0, hf_repo_name, epoch=0)
    config = Conf(**model["config"])
    del model

    iters = (
        list(range(0, 200, 10))
        + list(range(200, 1200, 20))
        + list(range(1200, 5_001, 50))
        + list(range(5_100, 10_001, 100))
        + list(range(10_500, 25_501, 500))
        + list(range(26_000, 100_501, 1000))
    )


    # Call the main function with the appropriate environment
    main(iters, hf_repo_name, config, run_on, use_wandb=True)
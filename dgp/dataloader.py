import torch
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
import os
from .PCSG import PCSG
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import TensorDataset
from typing import Tuple


def get_dataloader_and_vocab_size(
    config,
    preloaded_batches,
    preloader_kwargs=dict(num_batches=1000, make_masks=True, shuffle=True),
    eval=True,
) -> Tuple[DataLoader, int]:
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
        evaluation=eval,
    )
    if preloaded_batches:
        return get_preloaded_dataloader(
            config=config, dataloader=dataloader, **preloader_kwargs, eval=eval
        ), dataloader.dataset.PCSG.vocab_size

    return dataloader, dataloader.dataset.PCSG.vocab_size


def make_preloaded_dataset(
    dataloader: DataLoader,
    num_batches: int = 1000,
    eval: bool = True,
    make_masks: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_token_id = dataloader.dataset.pad_token_id
    all_inputs = []
    all_labels = []
    all_masks = []
    for i, batch in tqdm(enumerate(dataloader)):
        if i >= num_batches:
            break
        if not eval:
            sequences = batch[0]
            inputs = sequences[:, :-1]
            labels = sequences[:, 1:].clone()
            # procesed_batch = [inputs, labels]
            all_inputs.append(inputs)
            all_labels.append(labels)

            mask = (labels != pad_token_id).float()
            all_masks.append(mask)
        else:
            i, l, m = batch
            all_inputs.append(i)
            all_labels.append(l)
            all_masks.append(m)
    
    all_inputs = torch.cat(all_inputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    if not make_masks:
        labels[labels == pad_token_id] = -100
        return TensorDataset(all_inputs, all_labels)

    return TensorDataset(all_inputs, all_labels, all_masks)


def get_preloaded_dataloader(
    config, dataloader, num_batches=1000, make_masks=True, eval=True, shuffle=True
) -> DataLoader:
    # make a dataset with all the inputs and labels
    print("Making a new dataloader with all the data")
    dataset = make_preloaded_dataset(
        dataloader,
        num_batches=num_batches,
        eval=eval,
        make_masks=make_masks
    )
    new_dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=shuffle,
    )
    print("Done making the new dataloader")
    return new_dataloader


def get_dataloader(
    n_relative_properties: int = 25,
    n_descriptive_properties: int = 50,
    n_descriptive_values: int = 25,
    num_of_classes_to_divide_over: int = 3,
    prior_param: float = 5e-2,
    props_prior_type: str = "dirichlet",
    n_entities: int = 25,
    instr_ratio: float = 0.0,
    max_sample_length: int = 128,
    num_iters: int = 1e6,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    evaluation: bool = False,
):
    """
    Get the PCSG dataloader.

    Args:
        depth: int, depth of the PCSG
        n_relative_properties: int, number of relative properties
        n_descriptive_properties: int, number of descriptive properties
        n_descriptive_values: int, maximum number of descriptive values
        num_of_classes_to_divide_over: int, number of classes to divide over
        n_entities: int, number of non-PII values
        instr_ratio: float, ratio of instruction tasks
        max_sample_length: int, maximum sequence length
        num_iters: int, number of iterations
        batch_size: int, batch size
        num_workers: int, number of workers

    Returns:
        dataloader: torch.utils.data.DataLoader, PCSG dataloader
    """

    # Create a dataset
    dataset = PCSGDataset(
        n_relative_properties=n_relative_properties,
        n_descriptive_properties=n_descriptive_properties,
        n_descriptive_values=n_descriptive_values,
        num_of_classes_to_divide_over=num_of_classes_to_divide_over,
        prior_param=prior_param,
        props_prior_type=props_prior_type,
        n_entities=n_entities,
        instr_ratio=instr_ratio,
        num_iters=num_iters,
        max_sample_length=max_sample_length,
        seed=seed,
        evaluation=evaluation,
    )

    # Create a dataloader
    dataloader = DataLoader(
        dataset,
        sampler=torch.utils.data.RandomSampler(dataset, replacement=True),
        shuffle=False,
        pin_memory=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return dataloader


class PCSGDataset:
    def __init__(
        self,
        n_relative_properties=96,
        n_descriptive_properties=360,
        n_descriptive_values=25,
        num_of_classes_to_divide_over=24,
        prior_param=5e-2,
        props_prior_type="dirichlet",
        n_entities=25,
        instr_ratio: float = 0.0,
        num_iters: int = 1e6,
        max_sample_length: int = 128,
        seed: int = 42,
        evaluation: bool = False,
    ):
        """
        Initialize the PCSG dataset.
        Args:
            depth: int, depth of the PCSG
            n_relative_properties: int, number of relative properties
            n_descriptive_properties: int, number of descriptive properties
            n_descriptive_values: int, maximum number of descriptive values
            num_of_classes_to_divide_over: int, number of classes to divide over
            n_entities: int, number of non-PII values
            instr_ratio: float, ratio of instruction tasks
            num_iters: int, number of iterations
            max_sample_length: int, maximum sequence length
        """

        # Some setup details
        self.num_iters = int(num_iters)
        self.instr_ratio = instr_ratio
        self.max_sample_length = max_sample_length

        # Instructions / tasks
        self.tasks_dict = {}
        for n_task, task in enumerate(["null", "unscramble", "conditional generation"]):
            self.tasks_dict[f"T{n_task}"] = task
        self.task_tokens = list(self.tasks_dict.keys())
        if self.instr_ratio > 0:
            n_nonnull_tasks = len(self.task_tokens) - 1
            self.prior_over_tasks = [1.0 - self.instr_ratio] + n_nonnull_tasks * [
                self.instr_ratio / n_nonnull_tasks
            ]
        else:
            self.prior_over_tasks = [1.0]

        # Instruction decorator
        self.instruction_decorator = "Task: {task_token} \n Ops: {ops} \n Out:"

        # Define the PCSG
        self.PCSG = PCSG(
            n_relative_properties=n_relative_properties,
            n_descriptive_properties=n_descriptive_properties,
            n_descriptive_values=n_descriptive_values,
            num_of_classes_to_divide_over=num_of_classes_to_divide_over,
            prior_param=prior_param,
            props_prior_type=props_prior_type,
            n_entities=n_entities,
            tasks=self.task_tokens,
            seed=seed,
        )

        # Special tokens stuff
        self.pad_token = "<pad>"
        self.pad_token_id = self.PCSG.vocab["<pad>"]

        # Tokenize the task tokens
        self.task_token_idx = {t: self.PCSG.tokenize_sentence(t)[0] for t in self.task_tokens}

        # Define the PCSG generator
        self.generator = self.PCSG.sentence_generator(num_of_samples=self.num_iters)

        # Generation input template
        self.template = torch.tensor(self.PCSG.tokenize_sentence("Task: T0 \n Ops: <null> \n Out:"))

        self.evaluation = evaluation

    def save_grammar(self, path_to_results: str):
        """
        Save the grammar underlying the dataset
        """
        base_dir = os.path.join(path_to_results, "grammar")
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "PCSG.pkl"), "wb") as f:
            pkl.dump(self.PCSG, f)

    def load_grammar(self, path_to_results: str):
        """
        Load and override grammar of the dataset
        """
        base_dir = os.path.join(path_to_results, "grammar")
        with open(os.path.join(base_dir, "PCSG.pkl"), "rb") as f:
            self.PCSG = pkl.load(f)

    def __len__(self):
        """
        Return the number of iterations made in the training loop per epoch.
        """
        return self.num_iters

    def __getitem__(self, index):
        """
        Get the next sequence from the PCSG generator.
        """

        while True:
            # Generate a sequence from the PCSG
            sequence, symb_sequence, seq_logprob, conditioning_info = self.generator.__next__()

            # Instruction decorator
            task_token = np.random.choice(self.task_tokens, p=self.prior_over_tasks)

            # Define operands
            if task_token == "T0":  # Null task: generate valid sentences
                ops = "<null>"

            elif task_token == "T1":  # Unscramble task: unscramble the sentence
                ops = sequence.split()[:-1]
                np.random.shuffle(ops)
                ops = " ".join(ops)

            elif (
                task_token == "T2"
            ):  # Conditional generation task: generate a sentence conditioned on some vars
                cond_vars, all_vars = [], []

                # Subject identifiers
                for phrase_id in conditioning_info.keys():
                    subjects = conditioning_info[phrase_id]["subjects_idx"]
                    if len(subjects) > 0:
                        n_subjects = 1 + np.random.randint(len(subjects))
                        cond_vars.append(
                            list(np.random.choice(subjects, size=n_subjects, replace=False))
                        )

                    # Object identifiers
                    objects = conditioning_info[phrase_id]["objects_idx"]
                    if len(objects) > 0:
                        n_objects = 1 + np.random.randint(len(objects))
                        cond_vars.append(
                            list(np.random.choice(objects, size=n_objects, replace=False))
                        )

                    # Verbs
                    verbs = conditioning_info[phrase_id]["verbs"]
                    if len(verbs) > 0:
                        n_verbs = 1 + np.random.randint(len(verbs))
                        cond_vars.append(list(np.random.choice(verbs, size=n_verbs, replace=False)))

                    # Descriptors
                    descriptors = conditioning_info[phrase_id]["descriptors"]
                    if len(descriptors) > 0:
                        n_descriptors = 1 + np.random.randint(len(descriptors))
                        cond_vars.append(
                            list(np.random.choice(descriptors, size=n_descriptors, replace=False))
                        )

                # Flatten
                if len(cond_vars) > 0:
                    np.random.shuffle(cond_vars)
                    cond_vars = [v[0] for v in cond_vars]
                    ops = " ".join(cond_vars)

                else:
                    task_token, ops = "T0", "<null>"

            # Define instruction decorator
            instr = self.instruction_decorator.format(task_token=task_token, ops=ops)

            # Tokenize the sequence
            instr = torch.tensor(self.PCSG.tokenize_sentence(instr))
            sequence = torch.tensor(self.PCSG.tokenize_sentence(sequence))
            seq_length = float(sequence.size(0))

            # Join instruction decorator and sequence
            sequence = torch.cat((instr, sequence))

            # Truncate the sequence if it is longer than the max sequence length
            if sequence.size(0) > self.max_sample_length - 10:
                # sequence = sequence[:self.max_sample_length]
                pass

            # Pad the sequence to the max sequence length with <pad>
            else:
                sequence = torch.cat(
                    (
                        sequence,
                        torch.tensor(
                            [self.pad_token_id] * (self.max_sample_length - len(sequence))
                        ),
                    )
                )
                break

        if not self.evaluation:
            return sequence, symb_sequence, seq_length, seq_logprob, 0

        inputs, labels = sequence[:-1], sequence[1:]
        labels = labels.clone()
        mask = torch.ones_like(inputs)
        mask[labels == self.pad_token_id] = 0
        return inputs, labels, mask

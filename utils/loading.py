import os
import re
from dataclasses import dataclass

import huggingface_hub
import torch
import yaml


def load_model_for_iteration(
    it, dirname, epoch=0, device="cuda" if torch.cuda.is_available() else "cpu"
):
    fname = f"ckpt_epoch_{epoch}_iter_{it}.pt"
    return torch.load(f"{dirname}/{fname}", map_location=device)


def load_model_from_hf(
    it,
    repo_name,
    epoch=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    branch="main",
    repo_type="model",
    local_dir="./cache",
    clear_cache=True,
):
    file_name = f"ckpt_epoch_{epoch}_iter_{it}.pt"
    file = huggingface_hub.hf_hub_download(
        repo_id=repo_name,
        repo_type=repo_type,
        filename=file_name,
        revision=branch,
        local_dir=local_dir,
    )
    model = load_model_for_iteration(it, dirname=local_dir, epoch=epoch, device=device)
    if clear_cache:
        os.remove(file)
    return model


@dataclass
class Conf:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, Conf(**v))
            else:
                setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def load_from_yaml(cls, config_file):
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )

        conf_yaml = yaml.load(open(config_file), Loader=loader)

        return cls(**conf_yaml)

    @classmethod
    def load_from_dict(cls, conf_dict):
        return cls(**conf_dict)

    def to_dict(self):
        dict_conf = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Conf):
                dict_conf[k] = v.to_dict()
            else:
                dict_conf[k] = v
        return dict_conf

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def to_yaml(self, filename):
        with open(filename, "w") as f:
            yaml.dump(self.to_dict(), f)

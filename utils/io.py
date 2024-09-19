import re
from dataclasses import dataclass

import torch
import yaml


def load_model_for_iteration(
    it, dirname, epoch=0, device="cuda" if torch.cuda.is_available() else "cpu"
):
    fname = f"ckpt_epoch_{epoch}_iter_{it}.pt"
    return torch.load(f"{dirname}/{fname}", map_location=device)


@dataclass
class Conf:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, Conf(**v))
            else:
                setattr(self, k, v)

    def __repr__(self):
        yaml_str = yaml.dump(self.__dict__)
        return yaml_str

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def load_from_yaml(config_file):
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

        return Conf(**conf_yaml)

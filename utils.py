import importlib.util
import os

import numpy as np
import torch
import torch.utils.benchmark as benchmark
import trimesh
from torch.optim.lr_scheduler import StepLR, ConstantLR, ReduceLROnPlateau, MultiStepLR
from torchsummary import summary

from models.deep_sdf_decoder import Decoder

def load_checkpoint(path, model=None, optimizer=None, lr_schedule=None):
    data = torch.load(path)
    if model is not None:
        model.load_state_dict(data["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer_state_dict"])

    if lr_schedule is not None:
        lr_schedule.load_state_dict(data["lr_schedule"])


def pyvista2trimesh(pyvista_mesh):
    return trimesh.Trimesh(pyvista_mesh.points, pyvista_mesh.faces.reshape(-1, 4)[:, 1:])


def make_grid(v_min=-1.0, v_max=1.0, resolution=50j):
    X, Y, Z = np.mgrid[v_min:v_max:resolution, v_min:v_max:resolution, v_min:v_max:resolution]
    return np.stack((X, Y, Z), axis=3).reshape(-1, 3)


def get_learning_rate_schedules(schedule_specs, optimizer):
    # base on https://github.com/facebookresearch/DeepSDF

    schedules = []
    verbose = False
    for schedule_spec in schedule_specs:

        if schedule_spec.type == "Step":
            schedules.append(
                StepLR(optimizer, schedule_spec.step_size, schedule_spec.gamma, verbose=verbose)
            )
        elif schedule_spec.type == "ReduceLROnPlateau":
            schedules.append(
                ReduceLROnPlateau(optimizer, "min",
                                  factor=schedule_spec.factor, patience=schedule_spec.patience, verbose=verbose)
            )
        elif schedule_spec.type == "Constant":
            schedules.append(ConstantLR(optimizer, schedule_spec.factor, schedule_spec.total_iters, verbose=verbose))
        elif schedule_spec.type == "MultiStepLR":
            schedules.append(MultiStepLR(optimizer, milestones=schedule_spec.milestones,
                                         gamma=schedule_spec.gamma, verbose=verbose))
        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(schedule_spec.type)
            )

    return schedules


def summery_model(model):
    summary(model, (3,), device="cuda")


def __run_benchmark(model, x):
    return model(x)


def benchmark_model(model, batch_size, is_cuda=False):
    x = torch.rand(batch_size, 3)
    model.eval()

    if is_cuda:
        x = x.cuda()
        model = model.cuda()

    t_model = benchmark.Timer(
        stmt='__run_benchmark(model, x)',
        setup='from utils import __run_benchmark',
        globals={'model': model, 'x': x}
    )

    m = t_model.blocked_autorange()
    print("Mean batch time: {:6.2f} ms".format(m.mean * 1e3))
    print("Mean time by sample: {:6.2f} us".format(m.mean * 1e6 / batch_size))


def make_preload_model(path):
    save_dir = os.path.dirname(path)

    spec = importlib.util.spec_from_file_location("config", os.path.join(save_dir, "deep_sdf_config.py"))
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

    model = Decoder(0, foo.config.dense_layers,
                    dropout=foo.config.dropout,
                    dropout_prob=foo.config.dropout_prob,
                    norm_layers=foo.config.norm_layers,
                    latent_in=foo.config.latent_in,
                    xyz_in_all=foo.config.xyz_in_all,
                    use_tanh=foo.config.use_tanh,
                    latent_dropout=foo.config.latent_dropout,
                    weight_norm=foo.config.weight_norm,
                    nonlinearity=foo.config.get("nonlinearity", "relu"))
    load_checkpoint(path, model)
    model = model.cuda()
    return model

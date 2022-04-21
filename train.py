import os
import shutil

import torch
from mesh_to_sdf import scale_to_unit_sphere
from torch.utils.data import DataLoader

from config.deep_sdf_config import config
from dataset import SDFSamples
from metric import F1_metric
from models.deep_sdf_decoder import Decoder
from utils import get_learning_rate_schedules, load_checkpoint

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
experiment_directory = r""

load_path = None


def __save(file_name, epoch, model, optimizer, lr_schedule, loss_value, f1_score=None):
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    params_dir = os.path.join(experiment_directory, file_name)

    torch.save(
        {"epoch": epoch,
         "l1_loss": loss_value,
         "f1_score": f1_score,
         "optimizer_state_dict": optimizer.state_dict(),
         "model_state_dict": model.state_dict(),
         "lr_schedule": lr_schedule.state_dict()
         }, params_dir)


def save_best(epoch, model, optimizer, lr_schedule, loss_value, f1_score=None):
    __save("best.pth", epoch, model, optimizer, lr_schedule, loss_value, f1_score)


def save_latest(epoch, model, optimizer, lr_schedule, loss_value, f1_score=None):
    __save("best.pth", epoch, model, optimizer, lr_schedule, loss_value, f1_score)


def save_checkpoints(epoch, model, optimizer, lr_schedule, loss_value, f1_score=None):
    __save(str(epoch) + ".pth", epoch, model, optimizer, lr_schedule, loss_value, f1_score)



def copy_config_file(config_name):
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    shutil.copyfile(f"config/{config_name}.py", os.path.join(experiment_directory, f"{config_name}.py"))


def train(model, loss, dataloader, optimizer,
          epoch=0, enforce_minmax=False, clamp_dist=0.1):
    l1_func = torch.nn.L1Loss(reduction="sum")
    epoch_loss = 0.0
    l1_dict = 0.0
    for [points], [sdf_data] in dataloader:

        num_sdf_samples = sdf_data.shape[0]

        points = points.to(DEVICE)
        sdf_gt = sdf_data.unsqueeze(1).to(DEVICE)

        if enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        optimizer.zero_grad()

        pred_sdf = model(points)

        if enforce_minmax:
            pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        chunk_loss = loss(pred_sdf, sdf_gt)  # / num_sdf_samples
        chunk_loss.backward()

        epoch_loss += chunk_loss.item()  # * num_sdf_samples

        optimizer.step()

        with torch.no_grad():
            l1_dict += l1_func(pred_sdf, sdf_gt)

    epoch_loss = epoch_loss / (len(dataloader) * dataloader.dataset.batch_size)
    l1_dict = l1_dict / (len(dataloader) * dataloader.dataset.batch_size)
    print(f"Epoch {epoch}: Total loss {epoch_loss:.4e}, L1 {l1_dict:.6f}")

    return epoch_loss


def validation(mesh, decoder, cache_path=None):
    return F1_metric(decoder, mesh, cache_path=cache_path)


def main(mesh, mesh_name):
    global DEVICE

    DEVICE = torch.device('cpu')
    if config.gpu and torch.cuda.is_available():
        DEVICE = torch.device('cuda')

    copy_config_file("deep_sdf_config")

    decoder = Decoder(0, config.dense_layers,
                      dropout=config.dropout,
                      dropout_prob=config.dropout_prob,
                      norm_layers=config.norm_layers,
                      latent_in=config.latent_in,
                      xyz_in_all=config.xyz_in_all,
                      use_tanh=config.use_tanh,
                      latent_dropout=config.latent_dropout,
                      weight_norm=config.weight_norm,
                      nonlinearity=config.nonlinearity)
    decoder = decoder.to(DEVICE)

    loss = torch.nn.L1Loss(reduction="sum")
    # loss = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(decoder.parameters(), config.lr, weight_decay=config.weight_decay)
    lr_schedule = get_learning_rate_schedules(config.schedule_specs, optimizer)[0]

    sdf_dataset = SDFSamples(mesh, batch_size=config.batch_size, num_points=config.num_points, mesh_name=mesh_name)
    sdf_loader = DataLoader(sdf_dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)

    if load_path is not None:
        load_checkpoint(load_path, decoder, optimizer)

    best_loss = 100.0
    for epoch in range(config.epoch_num):

        loss_value = train(decoder, loss, sdf_loader, optimizer,
                           epoch=epoch, enforce_minmax=config.enforce_minmax,
                           clamp_dist=config.clamp_dist)
        if (epoch + 1) % config.saving_step == 0:
            f1_score = validation(mesh, decoder, f"{mesh_name}.pkl")
            save_checkpoints(epoch, decoder, optimizer, lr_schedule, loss_value, f1_score)
            print("F1={:.4f}".format(f1_score))

        if best_loss > loss_value:
            save_best(epoch, decoder, optimizer, lr_schedule, loss_value)
            best_loss = loss_value

        lr_schedule.step()

    f1_score = validation(mesh, decoder, f"{mesh_name}.pkl")
    save_latest(epoch, decoder, optimizer, lr_schedule, loss_value, f1_score)


if __name__ == '__main__':
    import trimesh
    import datetime

    from pyvista import examples
    from utils import pyvista2trimesh


    for mesh, mesh_name in [
        (pyvista2trimesh(examples.download_bunny_coarse()), "bunny_coarse"),
        (pyvista2trimesh(examples.download_dragon()), "dragon"),
        (trimesh.load(r'meshes/plane.obj'), "plane"),
        (trimesh.load(r'meshes/chair.obj'), "chair"),
        (trimesh.load(r'meshes/lamp.obj'), "lamp"),
        (trimesh.load(r'meshes/sofa.obj'), "sofa"),
        (trimesh.load(r'meshes/table.obj'), "table"),
    ]:
        experiment_directory = r"checkpoints/{}".format(mesh_name)

        gt_mesh = scale_to_unit_sphere(mesh)
        t1 = datetime.datetime.now()
        main(mesh, mesh_name)
        print(datetime.datetime.now() - t1)


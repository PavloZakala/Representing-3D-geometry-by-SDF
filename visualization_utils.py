import numpy as np
import skimage
import torch
import torch.utils.benchmark as benchmark
import trimesh
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from mesh_to_sdf import scale_to_unit_sphere, mesh_to_sdf
from torchsummary import summary

from utils import make_grid


def show_slice_window(data):
    size = data.shape[0]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)
    ax.imshow(data[:, 0])

    ax_layer = plt.axes([0.25, 0.2, 0.65, 0.03])
    layer_slider = Slider(ax_layer, 'Layer', 0.0, float(size) - 1, 0.0)

    def update(val):
        v = int(layer_slider.val)
        ax.imshow(data[:, v])

    layer_slider.on_changed(update)

    plt.show(block=True)


def visualize_sdf_surfaces(model, gt_mesh, resolution=64, is_color_grad=True):
    points = make_grid(-1.0, 1.0, complex(0, resolution))
    with torch.no_grad():
        points = torch.from_numpy(points.astype(np.float32)).cuda()
        predict_sdf = np.concatenate([model(batch).cpu().numpy() for batch in torch.chunk(points, 1024 * 16)])

    predict_voxels = predict_sdf.reshape(resolution, resolution, resolution)

    vertices, faces, normals, _ = skimage.measure.marching_cubes(predict_voxels, level=0.0)
    pred_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

    if is_color_grad:
        pred_mesh = scale_to_unit_sphere(pred_mesh)

        gt_sdf = mesh_to_sdf(gt_mesh, pred_mesh.vertices)

        with torch.no_grad():
            pred_vertices_torch = torch.from_numpy(pred_mesh.vertices.astype(np.float32)).cuda()
            predict_vertices_sdf = np.concatenate(
                [model(batch).cpu().numpy() for batch in torch.chunk(pred_vertices_torch, 1024 * 16)])

        diff = gt_sdf - predict_vertices_sdf[:, 0]
        diff = (diff - diff.min()) / (diff.max() - diff.min()) - 0.5

        vertic_colors = np.ones_like(pred_mesh.vertices) * 0.5
        vertic_colors[diff > 0] += np.stack([diff, -diff, -diff], axis=1)[diff > 0]
        vertic_colors[diff < 0] += np.stack([diff, diff, -diff], axis=1)[diff < 0]

        pred_mesh = trimesh.Trimesh(vertices=pred_mesh.vertices, faces=pred_mesh.faces, vertex_colors=vertic_colors)

    pred_mesh.show()
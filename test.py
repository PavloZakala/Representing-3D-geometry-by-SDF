import trimesh

from mesh_to_sdf import scale_to_unit_sphere
from pyvista import examples


from metric import F1_metric
from utils import pyvista2trimesh, make_preload_model
from visualization_utils import visualize_sdf_surfaces

MESHES = {
    "bunny_coarse": (pyvista2trimesh(examples.download_bunny_coarse()), r"checkpoints/bunny_coarse/latest.pth"),  # F1=0.8904
    "dragon": (pyvista2trimesh(examples.download_dragon()), r"checkpoints/dragon/3399.pth"),                      # F1=0.7991
    "plane": (trimesh.load(r'meshes/plane.obj'), r"checkpoints/plane/latest.pth"),                                # F1=0.9165
    "chair": (trimesh.load(r'meshes/chair.obj'), r"checkpoints/chair/best.pth"),                                  # F1=0.8697
    "lamp": (trimesh.load(r'meshes/lamp.obj'), r"checkpoints/lamp/3399.pth"),                                     # F1=0.8830
    "sofa": (trimesh.load(r'meshes/sofa.obj'), r"checkpoints/sofa/3199.pth"),                                     # F1=0.9014
    "table": (trimesh.load(r'meshes/table.obj'), r"checkpoints/table/best.pth"),                                  # F1=0.9090
}

if __name__ == '__main__':

    for mesh_name, (gt_mesh, checkpoint_path) in MESHES.items():
        print(f"Model {mesh_name}")
        gt_mesh = scale_to_unit_sphere(gt_mesh)

        decoder = make_preload_model(checkpoint_path)
        print("F1={:.4f}".format(F1_metric(decoder, gt_mesh, cache_path=f"{mesh_name}.pkl")))

        visualize_sdf_surfaces(decoder, gt_mesh, resolution=200, is_color_grad=True)

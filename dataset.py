import os
import pickle as pkl

import numpy as np
from torch.utils.data.dataset import Dataset
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere
from mesh_to_sdf import scale_to_unit_sphere, get_surface_point_cloud, BadMeshException


def pc_sample_sdf_near_surface(self, number_of_points=500000, use_scans=True, sign_method='normal', normal_sample_count=11,
                            min_size=0, return_gradients=False):
    query_points = []
    surface_sample_count = int(number_of_points * 47 / 50) // 2
    surface_points = self.get_random_surface_points(surface_sample_count, use_scans=use_scans)
    query_points.append(surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3)))
    query_points.append(surface_points + np.random.normal(scale=0.00025, size=(surface_sample_count, 3)))

    unit_sphere_sample_count = number_of_points - surface_points.shape[0] * 2
    unit_sphere_points = sample_uniform_points_in_unit_sphere(unit_sphere_sample_count)
    query_points.append(unit_sphere_points)
    query_points = np.concatenate(query_points).astype(np.float32)

    if sign_method == 'normal':
        sdf = self.get_sdf_in_batches(query_points, use_depth_buffer=False, sample_count=normal_sample_count,
                                      return_gradients=return_gradients)
    elif sign_method == 'depth':
        sdf = self.get_sdf_in_batches(query_points, use_depth_buffer=True, return_gradients=return_gradients)
    else:
        raise ValueError('Unknown sign determination method: {:s}'.format(sign_method))
    if return_gradients:
        sdf, gradients = sdf

    if min_size > 0:
        model_size = np.count_nonzero(sdf[-unit_sphere_sample_count:] < 0) / unit_sphere_sample_count
        if model_size < min_size:
            raise BadMeshException()

    if return_gradients:
        return query_points, sdf, gradients
    else:
        return query_points, sdf

def sample_sdf_near_surface(mesh, number_of_points=500000, surface_point_method='scan', sign_method='normal',
                            scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11,
                            min_size=0, return_gradients=False):
    mesh = scale_to_unit_sphere(mesh)

    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'

    surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method, 1, scan_count, scan_resolution,
                                                  sample_point_count,
                                                  calculate_normals=sign_method == 'normal' or return_gradients)

    return pc_sample_sdf_near_surface(surface_point_cloud, number_of_points, surface_point_method == 'scan', sign_method,
                                                       normal_sample_count, min_size, return_gradients)


class SDFSamples(Dataset):

    def __init__(self, mesh, batch_size=1024, num_points=100000, mesh_name=""):

        self.mesh = mesh
        self.batch_size = batch_size
        self.cache_name = os.path.join("cache", f"{mesh_name}_data.pkl")

        if os.path.exists(self.cache_name):
            with open(self.cache_name, "rb") as f:
                data = pkl.load(f)
            self.points = data["points"]
            self.sdf_data = data["sdf"]

        else:
            self.points, self.sdf_data = sample_sdf_near_surface(self.mesh, number_of_points=num_points,
                                                                 scan_resolution=1600)
            with open(self.cache_name, "wb") as f:
                pkl.dump({"points": self.points,
                          "sdf": self.sdf_data}, f)

        assert len(self.sdf_data) == num_points
        indexes = np.arange(len(self.sdf_data))
        np.random.shuffle(indexes)

        self.points = self.points[indexes]
        self.sdf_data = self.sdf_data[indexes]

    def __len__(self):
        return len(self.sdf_data) // self.batch_size + 1

    def __getitem__(self, idx):
        st_idx = self.batch_size * idx
        end_idx = min(self.batch_size * (idx + 1), len(self.sdf_data))
        return self.points[st_idx: end_idx], self.sdf_data[st_idx: end_idx]

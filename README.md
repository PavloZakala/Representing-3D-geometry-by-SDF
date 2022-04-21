# Representing 3D-geometry with SDF

## Researched papers

Base: [DeepSDF](https://arxiv.org/pdf/1901.05103.pdf)

Other link: 
- https://arxiv.org/pdf/1802.05384.pdf
- https://arxiv.org/pdf/2006.09662.pdf
- https://arxiv.org/pdf/2006.09661.pdf
- https://stackoverflow.com/questions/68178747/fast-2d-signed-distance
- https://scikit-robot.readthedocs.io/en/latest/reference/sdfs.html
- https://pypi.org/project/mesh-to-sdf/

GitHub:
- https://github.com/facebookresearch/DeepSDF
- https://github.com/vsitzmann/metasdf
- https://github.com/vsitzmann/siren

## Experiments
        
Висновки з експериментів:
- Оптимальний batch_size (1024, 1024 * 16). Якщо менше, навчається занадто повільно, якщо більше - погано сходиться
- hidden_lauer_size = 256 оптимальний. Якщо 128, то F1 не більше 0.8
- Результат сильно залежться від деталізації даних. scan_resolution треба ставити більшим.
- L2 loss - добре збігання на початку, і швидке формування форми моделі, але слабка деталзація


## Results 

### Benchmarks

```
benchmarks.py

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                  [-1, 256]           1,024
              ReLU-2                  [-1, 256]               0
            Linear-3                  [-1, 253]          65,021
              ReLU-4                  [-1, 253]               0
            Linear-5                  [-1, 256]          65,792
              ReLU-6                  [-1, 256]               0
            Linear-7                  [-1, 253]          65,021
              ReLU-8                  [-1, 253]               0
            Linear-9                  [-1, 256]          65,792
             ReLU-10                  [-1, 256]               0
           Linear-11                  [-1, 253]          65,021
             ReLU-12                  [-1, 253]               0
           Linear-13                  [-1, 256]          65,792
             ReLU-14                  [-1, 256]               0
           Linear-15                  [-1, 256]          65,792
             ReLU-16                  [-1, 256]               0
           Linear-17                    [-1, 1]             257
             Tanh-18                    [-1, 1]               0
================================================================
Total params: 459,512
Trainable params: 459,512
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.03
Params size (MB): 1.75
Estimated Total Size (MB): 1.78
----------------------------------------------------------------
Mean batch time:   4.24 ms
Mean time by sample:   0.26 us
```




### bunny_coarse 
<img src="https://github.com/PavloZakala/Representing-3D-geometry-by-SDF/blob/main/images/bunny_coarse.jpg?raw=true" alt="target_heatmap">

#### test.py:   F1:0.8904

<img src="https://github.com/PavloZakala/Representing-3D-geometry-by-SDF/blob/main/images/dragon.jpg?raw=true" alt="target_heatmap">

#### test.py:   F1=0.7991

<img src="https://github.com/PavloZakala/Representing-3D-geometry-by-SDF/blob/main/images/plane.jpg?raw=true" alt="target_heatmap">

#### test.py:   F1=0.9165

<img src="https://github.com/PavloZakala/Representing-3D-geometry-by-SDF/blob/main/images/chair.jpg?raw=true" alt="target_heatmap">

#### test.py:   F1=0.8697 (There is a bug of visualization)

<img src="https://github.com/PavloZakala/Representing-3D-geometry-by-SDF/blob/main/images/lamp.jpg?raw=true" alt="target_heatmap">

#### test.py:   F1=0.8830 (There is a bug of visualization)

<img src="https://github.com/PavloZakala/Representing-3D-geometry-by-SDF/blob/main/images/sofa.jpg?raw=true" alt="target_heatmap">

#### test.py:   F1=0.9014

<img src="https://github.com/PavloZakala/Representing-3D-geometry-by-SDF/blob/main/images/table.jpg?raw=true" alt="target_heatmap">

#### test.py:   F1=0.9090

Tasks: 
1. How to visualize 3d plot 
    https://github.com/pyvista/pyvista
    https://docs.pyvista.org/examples/00-load/read-file.html
   
1.1 Research 3d PolyData (vertices, faces, lines)
    https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyData.html

2. Research SDF
    https://stackoverflow.com/questions/68178747/fast-2d-signed-distance
    https://scikit-robot.readthedocs.io/en/latest/reference/sdfs.html
    INSTALL (pip install scikit-robot) NOT (pip install skrobot)
   
    https://pypi.org/project/mesh-to-sdf/
    
    DeepSDF:
    https://medium.com/syncedreview/facebook-mit-uw-introduce-deepsdf-ai-for-3d-shape-representation-75416481e1be 
   
    AtlasNet and OGN
   

Benchmark
https://pytorch.org/tutorials/recipes/recipes/benchmark.html


https://arxiv.org/pdf/1802.05384.pdf
https://arxiv.org/pdf/1901.05103.pdf
https://arxiv.org/pdf/2006.09662.pdf

from easydict import EasyDict

config = EasyDict()

config.gpu = True
config.epoch_num = 3500
# config.epoch_num = 10
config.num_points = 500000
config.mesh_path = ""
config.batch_size = 1024 * 4
# config.batch_size = 1024
# config.batch_size = 256
config.num_workers = 0
config.lr = 1e-4

config.saving_step = 200
config.validation_step = 200

# Regularization
config.weight_decay = 0 #1e-8

# Model
# config.dense_layers = [512, 512, 512, 512, 512, 512, 512, 512]
config.dense_layers = [256, 256, 256, 256, 256, 256, 256, 256]
# config.dense_layers = [128, 128, 128, 128, 128, 128, 128, 128]
config.dropout = [0, 1, 2, 3, 4, 5, 6, 7],
config.dropout_prob = 0.2
config.norm_layers = [0, 1, 2, 3, 4, 5, 6, 7]
config.latent_in = [2, 4]
config.xyz_in_all = False
config.use_tanh = False
config.latent_dropout = False
config.weight_norm = True
config.nonlinearity="relu"

config.schedule_specs = [
    EasyDict({
        "type": "Step",
        "step_size": 600,
        "gamma": 0.5
    }),
    # EasyDict({
    #     "type": "MultiStepLR",
    #     "milestones": [200, 600, 1000, 1500],
    #     "gamma": 0.5
    # }),
    # EasyDict({
    #     "type": "ReduceLROnPlateau",
    #     "factor": 0.1,
    #     "patience": 5
    # }),
    # EasyDict({
    #     "type": "Constant",
    #     "factor": 0.5,
    #     "total_iters": 4
    # }),
]

# Clamp
config.enforce_minmax = True
config.clamp_dist = 0.1

import types

def get_args_parser():
    args = types.SimpleNamespace()
    # ---------------------------- Path ---------------------------- #
    args.data_name = 'depth'
    args.save_path = 'artifacts'

    # ----------------------- Data Paramters ----------------------- #
    args.batch_size = 2
    args.scale = 3
    args.height = 480
    args.width = 640
    args.train_scene_limit = 2000
    args.test_scene_limit = 500

    # ---------------------- Train Parameters ---------------------- #
    args.n_epoch = 50
    args.seed = 700
    args.lr = 1e-3
    args.weight_decay = 1e-4
    args.clip_max_norm = 0.0
    args.dist_thresh = 5
    args.loss_weights = [1., 1.]
    args.log_interval = 25
    args.save_interval = 1
    args.distributed = False

    # ----------------------- Model Patameters ---------------------- #
    args.d_coarse_model = 256
    args.d_fine_model = 128
    args.n_coarse_layers = 4
    args.n_finr_layers = 1
    args.n_heads = 8
    args.backbone_name = 'resnet34'
    args.matching_name = 'dual_softmax'
    args.match_threshold = 0.2
    args.window_size = 5
    args.border = 1
    args.load = None
    args.sinkhorn_iterations = 100

    # --------------- Distributed Training Parameters --------------- #
    args.world_size = 1
    args.dist_url = 'env://'

    return args
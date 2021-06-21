from models.model import MatchingNet

def build_model(args):
    return MatchingNet(
        args.d_coarse_model,
        args.d_fine_model,
        args.n_coarse_layers,
        args.n_fine_layers,
        args.n_heads,
        args.backbone_name,
        args.matching_name,
        args.match_threshold,
        args.window_size,
        args.border,
        args.sinkhorn_iterations,
    )
from datasets.depth_dataset import build_depth

def build_dataset(args):
    if args.data_name == 'depth':
        """
        json_path: Your own json path. Could create the json file by
        DISK[DISK:Learning local features with policy gradient.NeurIPS, 2020] ways
        """
        json_path = 'train/dataset.json' 
        return build_depth(
                json_path, 
                (args.height, args.width),
                args.scale,
                use_bins=(args.matching_name == 'sinkhorn'), 
                limit=args.train_scene_limit, 
                shuffle=False, 
                warn=True
        )
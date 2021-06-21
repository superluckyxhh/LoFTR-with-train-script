from loss.criterion import MatchingCriterion

def build_criterion(args):
    eps = 1e-10
    return MatchingCriterion( 
                args.data_name,
                args.matching_name,
                args.dist_thresh,
                args.loss_weights, 
                eps=eps)
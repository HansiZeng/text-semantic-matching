import torch

def get_mask(tensor, padding_idx=0):
    """
    Get a mask to `tensor`.
    Args:
        tensor: LongTensor with shape of [bz, seq_len]

    Returns:
        mask: BoolTensor with shape of [bz, seq_len]
    """
    mask = torch.ones(size=list(tensor.size()), dtype=torch.bool)
    mask[tensor[:,:] == padding_idx] = False 

    return mask 


def correct_instance_count(pred_logits, labels):
    """
    Args:
        pred_logits: FloatTensor with shape of [bz, number_of_classes_in_labels]
        labels: LongTensor with shape of [bz]
    
    Returns:
        correct_count: int
    """
    pred_labels = pred_logits.argmax(dim=1)

    return (pred_labels == labels).sum().item()

class Args():
    pass 

def parse_args(config):
    args = Args()
    with open(config, 'r') as f:
        config = json.load(f)
    for name, val in config.items():
        setattr(args, name, val)

    return args
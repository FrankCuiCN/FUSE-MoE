import torch


def packing(input_tensor, padding_size, mapping):
    """
    Explanation: Apply the same padding and mapping to the num_token dimension
    In:
      - input_tensor: (1, num_head, num_token, emb_size)
      - padding_size: int
      - mapping:      (num_token + padding_size,)
    Out:
      - input_tensor: (1, num_head, num_token + padding_size, emb_size)
    """
    # Set up variables
    num_head   = input_tensor.size(1)
    num_token  = input_tensor.size(2)
    emb_size   = input_tensor.size(3)
    # Get padding_tensor
    # (1, num_head, padding_size, emb_size)
    padding_tensor = torch.zeros((1, num_head, padding_size, emb_size), dtype=input_tensor.dtype, device=input_tensor.device)
    # Apply padding_tensor to input_tensor
    # (1, num_head, num_token + padding_size, emb_size)
    input_tensor = torch.concat((input_tensor, padding_tensor), dim=2)
    # Permute input_tensor
    # (1, num_head, num_token + padding_size, emb_size)
    input_tensor = input_tensor[:, :, mapping]
    return input_tensor

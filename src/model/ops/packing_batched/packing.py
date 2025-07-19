import torch


def packing(input_tensor, padding_size, mapping):
    """
    input_tensor: (batch_size, num_head, num_token, emb_size)
    padding_size: int
    mapping:      (batch_size * num_head * (num_token + padding_size),)
    output shape: (batch_size, num_head, num_token + padding_size, emb_size)
    """
    # Set up variables
    batch_size = input_tensor.size(0)
    num_head   = input_tensor.size(1)
    num_token  = input_tensor.size(2)
    emb_size   = input_tensor.size(3)
    # Flatten input_tensor
    # (batch_size * num_head * num_token, emb_size)
    input_tensor = input_tensor.view(batch_size * num_head * num_token, emb_size)
    # Get padding_tensor
    # (batch_size * num_head * padding_size, emb_size)
    padding_tensor = torch.zeros((batch_size * num_head * padding_size, emb_size), dtype=input_tensor.dtype, device=input_tensor.device)
    # Apply padding_tensor to input_tensor
    # (batch_size * num_head * num_token + batch_size * num_head * padding_size, emb_size)
    input_tensor = torch.concat((input_tensor, padding_tensor), dim=0)
    # Permute input_tensor
    input_tensor = input_tensor[mapping]
    # Unflatten input_tensor
    input_tensor = input_tensor.view(batch_size, num_head, num_token + padding_size, emb_size)
    return input_tensor

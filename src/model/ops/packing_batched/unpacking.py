def unpacking(input_tensor, padding_size, mapping_inv):
    """
    input_tensor: (batch_size, num_head, num_token + padding_size, emb_size)
    padding_size: int
    mapping_inv:  (batch_size * num_head * (num_token + padding_size),)
    output shape: (batch_size, num_head, num_token, emb_size)
    """
    # Set up variables
    batch_size = input_tensor.size(0)
    num_head = input_tensor.size(1)
    num_token = input_tensor.size(2) - padding_size
    emb_size = input_tensor.size(3)
    # Flatten input_tensor
    # (batch_size * num_head * num_token + batch_size * num_head * padding_size, emb_size)
    input_tensor = input_tensor.view(-1, emb_size)
    # Permute input_tensor
    input_tensor = input_tensor[mapping_inv]
    # Remove padding
    # Note: When k == 0, input_tensor[:-k] becomes empty
    input_tensor = input_tensor[:batch_size * num_head * num_token]
    # Unflatten input_tensor
    input_tensor = input_tensor.view(batch_size, num_head, num_token, emb_size)
    return input_tensor

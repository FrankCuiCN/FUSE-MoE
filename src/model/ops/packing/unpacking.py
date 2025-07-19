def unpacking(input_tensor, padding_size, mapping_inv):
    """
    Explanation: Remove the padding and apply mapping_inv to the num_token dimension
    In:
      - input_tensor: (1, num_head, num_token + padding_size, emb_size)
      - padding_size: int
      - mapping_inv:  (num_token + padding_size,)
    Out:
      - input_tensor: (1, num_head, num_token, emb_size)
    """
    # Set up variables
    num_head = input_tensor.size(1)
    num_token = input_tensor.size(2) - padding_size
    emb_size = input_tensor.size(3)
    # Permute input_tensor
    # (1, num_head, num_token + padding_size, emb_size)
    input_tensor = input_tensor[:, :, mapping_inv]
    # Remove padding
    # Note: When k == 0, input_tensor[:-k] becomes empty
    # (1, num_head, num_token, emb_size)
    input_tensor = input_tensor[:, :, :num_token]
    return input_tensor

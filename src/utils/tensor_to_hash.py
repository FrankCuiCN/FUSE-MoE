import hashlib


def tensor_to_hash(tensor):
    """Converts a PyTorch tensor to a SHA-256 hash string"""
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()

import numpy as np

def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def attention_scores(Q, K, scale=True):
    """
    Compute scaled dot-product attention scores.
    Q: (n, d_k) — queries
    K: (m, d_k) — keys
    Returns: (n, m) — similarity scores
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T
    if scale:
        scores = scores / np.sqrt(d_k)
    return scores

def layer_norm(x, gamma=1.0, beta=0.0, epsilon=1e-5):
    """
    Layer normalization.
    x: (..., d) — input
    Returns: normalized x
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + epsilon) * gamma + beta
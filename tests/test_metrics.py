import jax
import numpy as np
from metrics import binary_cross_entropy

def test_bce():
    mock_params = {}
    mock_model = lambda x: x  # Simple identity model
    X = jax.random.normal(jax.random.PRNGKey(0), (5, 10))
    y = jnp.array([0, 1, 0, 1, 0])
    
    loss = binary_cross_entropy(mock_params, mock_model, X, y)
    assert isinstance(loss, float), "BCE should return scalar"
    assert loss > 0, "Loss should be positive"
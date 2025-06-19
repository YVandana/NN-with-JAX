import jax.numpy as jnp
import numpy as np
from sklearn.metrics import accuracy_score

def calculate_metrics(params, model, X, y):
    """Calculate metrics for numpy inputs"""
    # Convert to JAX arrays
    X_jax = jnp.array(X)
    y_jax = jnp.array(y)
    
    # Get predictions
    y_pred = model.apply({'params': params}, X_jax)
    y_pred_class = (y_pred > 0.5).astype(int)
    
    # Ensure shapes are correct
    y_true = np.ravel(y)
    y_pred_class = np.ravel(y_pred_class)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred_class),
        'loss': float(-jnp.mean(y_jax * jnp.log(y_pred + 1e-8) + 
                               (1-y_jax) * jnp.log(1-y_pred + 1e-8)))
    }
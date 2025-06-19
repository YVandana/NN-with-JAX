import jax
from model import MLP
from data_loader import SyntheticDataset
import optax
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from metrics import calculate_metrics


def compute_loss(params, X, y, model):
    """Binary cross-entropy loss function"""
    predictions = model.apply({'params': params}, X)
    return -jnp.mean(y * jnp.log(predictions + 1e-8) + (1 - y) * jnp.log(1 - predictions + 1e-8))

def train():
    # Initialize model
    model = MLP(hidden_dim=32, output_dim=1)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, 10)))['params']
    
    # Dataset and optimizer
    dataset = SyntheticDataset()
    val_dataset = SyntheticDataset(n_samples=200)  # Validation set
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)
    
    # Track metrics
    train_metrics = []
    
    for epoch in range(10):
        epoch_train_metrics = []
        epoch_val_metrics = []
        epoch_loss = 0
        count = 0
        
        # Training loop
        for X, y in dataset:
            X, y = jnp.array(X), jnp.array(y)
            loss, grads = jax.value_and_grad(compute_loss)(params, X, y, model)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            # Calculate training metrics
            metrics = calculate_metrics(params, model, np.array(X), np.array(y))
            epoch_train_metrics.append(metrics)
            epoch_loss += loss
            count += 1

        # Validation loop
        for X_val, y_val in val_dataset:
            val_metrics = calculate_metrics(params, model, np.array(X_val), np.array(y_val))
            epoch_val_metrics.append(val_metrics)
        
        # Aggregate metrics
        train_metrics.append({
            'epoch': epoch,
            'train_loss': np.mean([m['loss'] for m in epoch_train_metrics]),
            'train_accuracy': np.mean([m['accuracy'] for m in epoch_train_metrics]),
            'val_loss': np.mean([m['loss'] for m in epoch_val_metrics]),
            'val_accuracy': np.mean([m['accuracy'] for m in epoch_val_metrics])
        })
        
        print(f"Epoch {epoch} | "
              f"Train Loss: {train_metrics[-1]['train_loss']:.4f} | "
              f"Train Acc: {train_metrics[-1]['train_accuracy']:.2%} | "
              f"Val Acc: {train_metrics[-1]['val_accuracy']:.2%}")

    return train_metrics, params

if __name__ == "__main__":
    metrics, params = train()
    
    # Generate visualizations
    from visualize import plot_training, plot_predictions
    plot_training(metrics)
    plot_predictions(params, MLP(hidden_dim=32, output_dim=1), SyntheticDataset(n_samples=10))
    
    print("Training complete. Visualizations saved to:")
    print("- training_metrics.png")
    print("- predictions.png")
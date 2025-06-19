import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

def plot_training(metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot (now using 'train_loss')
    ax1.plot([m['epoch'] for m in metrics], [m['train_loss'] for m in metrics], label='Train')
    ax1.plot([m['epoch'] for m in metrics], [m['val_loss'] for m in metrics], label='Validation')
    ax1.set_title('Training/Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Accuracy plot (now using 'train_accuracy' and 'val_accuracy')
    ax2.plot([m['epoch'] for m in metrics], [m['train_accuracy'] for m in metrics], label='Train')
    ax2.plot([m['epoch'] for m in metrics], [m['val_accuracy'] for m in metrics], label='Validation')
    ax2.set_title('Training/Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(params, model, test_data):
 # Get first 10 samples - handle both tuple and non-tuple datasets
    samples = list(test_data)[:10] if hasattr(test_data, '__iter__') else [test_data[i] for i in range(10)]
    
    # Extract and flatten features and labels
    X_test = jnp.array([x.reshape(-1) if hasattr(x, 'reshape') else x for x, _ in samples])
    y_test = jnp.array([y for _, y in samples])
    
    # Make predictions and flatten
    preds = model.apply({'params': params}, X_test)
    preds = np.array(preds).flatten()  # Convert to numpy and flatten
    y_test = np.array(y_test).flatten()
    
    # Plot with error handling
    plt.figure(figsize=(10, 5))
    
    # Use proper numeric values for bar positions
    x_pos = np.arange(len(preds))
    bar_width = 0.4
    
    plt.bar(x_pos - bar_width/2, preds, width=bar_width, alpha=0.7, label='Predictions')
    plt.bar(x_pos + bar_width/2, y_test, width=bar_width, alpha=0.7, label='True Labels')
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.xticks(x_pos, [f"Sample {i+1}" for i in range(len(preds))])
    plt.legend()
    plt.title('Model Predictions vs True Labels')
    plt.ylim(0, 1.1)
    plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
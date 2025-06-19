# Neural Network Implementation with JAX

![training_metrics](https://github.com/user-attachments/assets/6b517035-2373-484c-ae81-f427d3ba48f5)
![predictions](https://github.com/user-attachments/assets/38f35f86-65f0-454c-85b6-992e47bbe39e)


A complete implementation of a neural network using JAX, featuring:
- Custom MLP architecture
- Training/validation metrics tracking
- Interactive visualizations
- Synthetic data generation

## 📊 Performance Results
### Final Epoch Metrics
| Metric | Training | Validation |
|--------|----------|------------|
| Loss   | 0.0335   | -          |
| Accuracy | 99.80% | 98.50% |

### Training Progress
```text
Epoch 0 | Train Loss: 0.4550 | Train Acc: 81.50% | Val Acc: 96.00%
Epoch 1 | Train Loss: 0.1815 | Train Acc: 96.80% | Val Acc: 98.00%
...
Epoch 9 | Train Loss: 0.0335 | Train Acc: 99.80% | Val Acc: 98.50%
```
![Successful Execution 1](https://github.com/user-attachments/assets/162f8337-8929-4fbc-950d-3861a514fd10)


## 🛠️ Technical Stack

- Core Framework: JAX 0.4.13
- Optimization: Optax 0.1.7
- Visualization: Matplotlib 3.7.1
- Metrics: scikit-learn 1.3.2

## 📂 Project Structure
src/
├── train.py          # Main training script
├── model.py          # MLP architecture
├── data_loader.py    # Synthetic dataset
├── metrics.py        # Accuracy/loss calculations
└── visualize.py      # Plot generation

## 🚀 Getting Started

1. Install Dependencies

```
pip install -r requirements.txt
```

2. Run Training

```
python src/train.py
```

3. View Results 
  - training_metrics.png: Loss/accuracy curves

  - predictions.png: Model predictions vs true labels

## 🔍 Key Features

- **Modular Design**: Separate components for easy modification
- **Reproducible**: Fixed random seeds
- **Visual Diagnostics**: Clear performance tracking

## 📈 Interpretation

- The model achieves 99.8% training accuracy and 98.5% validation accuracy
- Rapid convergence in first 3 epochs (see loss curve)
- Minimal overfitting (small train-val accuracy gap)

## 🤝 Contributing
Pull requests welcome! For major changes, please open an issue first.

📜 License
![MIT](https://choosealicense.com/licenses/mit/)


### Key Features of This README:
1. **Visual Integration**: Embeds your actual result images
2. **Performance Highlights**: Shows key metrics prominently
3. **Structured Layout**: Clear sections for setup, results, and technical details
4. **Reproducibility**: Includes exact package versions
5. **Professional Formatting**: Tables and code blocks for readability

To use:
1. Save as `README.md` in your project root
2. Commit to GitHub:
   ```bash
   git add README.md
   git commit -m "Add comprehensive project documentation"
   git push
   ```

# 🎬 Movie Revenue Classification with Transformer (Tabular Data)

This project predicts the **box office revenue class** (Low / Medium / High) of movies using a **Transformer-based model** trained on **tabular features** such as budget, popularity, release date, and more.

## 🔧 Features Used

- `scaled_budget`
- `encoded_language`
- `scaled_popularity`
- `encoded_release_year`
- `encoded_release_month`
- `encoded_release_dayofweek`
- `scaled_runtime`
- `scaled_vote_average`

Target column: `encoded_revenue_class`  →  
`0`: Low, `1`: Medium, `2`: High

## 🧠 Model: Tabular Transformer

- Built using `PyTorch`
- Architecture includes:
  - Input Linear Layer
  - Transformer Encoder Layers (multi-head self-attention)
  - Final Linear Layer → 3-Class Output
---
### 2. Train the Model
```bash
python scripts/train.py
```
- Trains on the provided dataset
- Evaluates model and prints test loss, accuracy, and classification report
- Saves best model to `models/best_model.pt`

### 3. Predict a Movie’s Revenue Class
```bash
python scripts/predict.py
```
- Predicts revenue class based on a manually defined movie input

## 📊 Evaluation Results

```
Test Loss: 0.6564, Accuracy: 0.6993


              precision    recall  f1-score   support

         Low       0.74      0.70      0.72       320
      Medium       0.56      0.61      0.59       320
        High       0.81      0.79      0.80       321

    accuracy                           0.70       961
   macro avg       0.71      0.70      0.70       961
weighted avg       0.71      0.70      0.70       961

---

# 🎬 Movie Revenue Prediction with Transformer on Tabular Data

## 📋 About the Project

### Project Purpose and Real-World Problem

The film industry is a multi-billion dollar market. For producers, studios, and investors, predicting a film's financial success beforehand is critically important. This project aims to predict movie revenue categories (Low, Medium, High) using machine learning and modern Transformer architecture.

**Problem Being Solved:**
- 🎯 Estimating potential returns before film investment
- 📊 Understanding which factors (budget, director, actors, genre) affect revenue
- 💡 Making strategic decision-making processes data-driven

### Why Transformer?

Traditionally, Transformers are used in NLP (Natural Language Processing). However, using Transformers on **tabular data** in this project has advantages:

- ✅ Ability to learn complex relationships between features (self-attention mechanism)
- ✅ Handling high-dimensional data
- ✅ Less dependency on feature engineering
- ✅ Strong generalization capability

---

## 🗂️ Dataset

The project uses the **TMDB 5000 Movies** dataset:
- 📁 `tmdb_5000_movies.csv` - Movie information
- 📁 `tmdb_5000_credits.csv` - Cast and crew information
- 📊 Total: **4,803 movies**

### Features Used

| Category | Features |
|----------|-----------|
| **Financial** | Budget (scaled & log transformed) |
| **Popularity** | Popularity score, vote average |
| **Temporal** | Year, month, day of week |
| **Content** | Genre combinations, keywords |
| **Crew** | Director, main cast, production company |
| **Derived** | Budget-genre combo, popularity-year combo |

---

## 🔧 Data Processing and Feature Engineering

### 1. Missing Value Management

```python
# Median for numerical values
movies_df['budget'] = movies_df['budget'].fillna(movies_df['budget'].median())
movies_df['runtime'] = movies_df['runtime'].fillna(movies_df['runtime'].median())

# Default values for categorical data
movies_df['original_language'] = movies_df['original_language'].fillna('en')
movies_df['genres'] = movies_df['genres'].fillna('[]')
```

**Why this approach?**
- Median is not affected by outliers
- Empty list as default value is appropriate for JSON format columns

### 2. JSON Parsing and Feature Extraction

```python
def parse_and_join_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        if isinstance(genres, list):
          names = sorted([g["name"] for g in genres])
          return "|".join(names) if names else "None"
    except:
        return "None"

movies_df["genre_group"] = movies_df["genres"].apply(parse_and_join_genres)
```

**Important:** JSON formatted data (genre, cast, crew) is parsed and converted to usable format.

### 3. Frequency-Based Features

```python
director_freq = movies_df["director"].value_counts().to_dict()
movies_df["director_freq"] = movies_df["director"].map(director_freq)
```

**Idea:** Famous directors' films generally generate more revenue. Frequency value quantifies this information for the model.

### 4. Interaction Features

```python
movies_df["budget_genre_combo"] = movies_df["scaled_budget"] * movies_df["encoded_genre_group"]
movies_df["popularity_year_combo"] = movies_df["scaled_popularity"] * movies_df["encoded_release_year"]
```

**Why interactions?** High-budget action films have different revenue potential compared to low-budget dramas. This allows the model to learn such relationships.

### 5. Target Variable Creation

```python
movies_df["revenue_class"] = pd.qcut(movies_df["revenue"], q=3, labels=["Low", "Medium", "High"])
```

**Quantile-based binning:** Creates balanced classes by dividing the dataset into 3 equal groups.

---

## 🧠 Model Architecture: Tabular Transformer

### Model Components

```python
class TabularTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_classes=3, num_heads=4, num_layers=2, dropout=0.1):
        super(TabularTransformer, self).__init__()
        
        # 1. Input Projection: Features are transformed to model dimension
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        # 2. Transformer Encoder: Feature relationships learned via self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Classifier: Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
```

### Critical Design Decisions

#### 1. Input Projection
```python
x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
x = self.input_projection(x)  # [batch_size, 1, model_dim]
```
**Why?** Transformer expects sequences. For tabular data, we treat each sample as a single-element sequence.

#### 2. Self-Attention Mechanism
The transformer encoder automatically learns relationships between features:
- Budget ↔ Revenue
- Director frequency ↔ Popularity
- Genre ↔ Year

#### 3. Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_dim` | 64 | Embedding dimension |
| `num_heads` | 4 | Multi-head attention |
| `num_layers` | 2 | Number of transformer layers |
| `dropout` | 0.1 | Overfitting prevention |

---

## 📊 Training and Performance

### Training Parameters

```python
n_epochs = 20
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
```

### Data Split
- **Training:** 60% (2,881 movies)
- **Validation:** 20% (961 movies)
- **Test:** 20% (961 movies)

### Results

```
Test Accuracy: 69.93%

Classification Report:
              precision    recall  f1-score   support
         Low       0.74      0.70      0.72       320
      Medium       0.56      0.61      0.59       320
        High       0.81      0.79      0.80       321
```

**Analysis:**
- ✅ **High** class is predicted best (F1: 0.80)
- ⚠️ **Medium** class is most challenging (F1: 0.59) - Transition zone between classes
- 🎯 Overall performance: **~70%** - A good baseline

---

## 🚀 Usage

### 1. Installation

```bash
pip install pandas numpy scikit-learn torch
```

### 2. Model Training

```python
# Load data
movies_df = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Train model
model = TabularTransformer(input_dim=X_train.shape[1])
# ... (training code)
```

### 3. Making Predictions

```python
# Prediction for a new movie
new_movie = {
    "scaled_budget": 0.43,
    "encoded_language": 3,
    "scaled_popularity": 0.2,
    # ... other features
}

predicted_class = predict_movie(model, new_movie_scaled[0])
print(f"Predicted revenue class: {predicted_class}")  # Output: "Low" / "Medium" / "High"
```

---

## 💡 Key Notes and Improvement Suggestions

### Strengths
✅ Use of modern Transformer architecture  
✅ Comprehensive feature engineering  
✅ Frequency-based features (director, actors)  
✅ Interaction features

### Areas for Improvement
🔧 **Hyperparameter tuning:** Grid search or Optuna can be used  
🔧 **More layers:** Model depth can be increased  
🔧 **Ensemble methods:** Transformer + XGBoost combination  
🔧 **Attention visualization:** To see which features are important

---

## 📚 Resources

- [TMDB Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [PyTorch Transformer Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)


---

## 👨‍💻 Developer

**Project Owner:** [Erenkyz](https://github.com/Erenkyz)

---

## 📄 License

This project is provided under the MIT license.

---

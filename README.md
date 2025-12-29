# 🐾 PetFinder Multimodal Adoption Prediction

A multimodal deep learning project predicting **pet adoption speed (5-class classification)** by combining **image**, **text sentiment**, and **tabular metadata** features.  
Built entirely in **TensorFlow/Keras** and designed for experimentation with multiple model architectures (v1–v5).

---

## 📘 Overview

This project explores how integrating diverse data modalities can improve predictive performance in real-world ML problems.  
Using the **PetFinder.my Adoption Prediction** dataset (via a Hugging Face/Kaggle mirror), each pet sample includes:

- **Images** (up to 3 photos per pet)  
- **Tabular attributes** (age, breed, gender, vaccination, fee, etc.)  
- **Sentiment analysis** (Google NLP outputs: `SentimentScore`, `SentimentMagnitude`)  

The target variable `AdoptionSpeed` ∈ {0, 1, 2, 3, 4} measures how quickly a pet was adopted.

---

## 🧠 Model Architecture

| Version | Description | Input Modalities | Notes |
|----------|--------------|------------------|-------|
| **v1_tabular** | Baseline fully connected NN | Tabular only | 3 dense layers, dropout, batch norm |
| **v2_multimodal** | Early multimodal fusion | Tabular + Stacked 3-image tensor | CNN + MLP concatenation |
| **v3_multimodal** | `TimeDistributed` CNN (shared weights) | Tabular + 3 separate images | Late fusion + global pooling |
| **v4_multimodal** | Improved regularization, weighted classes | Same as v3 | EarlyStopping + ReduceLROnPlateau |
| **v5_multimodal** | Final refined architecture | Same | Tuned dropout + learning schedule |

All experiments use **stratified splits (80/10/10)**, **class-balanced weighting**, and **random seed = 842023** for reproducibility.

---

## ⚙️ Pipeline

1. **Data Preparation (`data_creator.py`)**  
   - Merges `train.csv` with `train_sentiment.csv`  
   - Filters rows with ≥ 3 photos and valid sentiment  
   - Builds stratified 2 500-row subset to balance rare class 0  

2. **Preprocessing (`ColumnTransformer`)**  
   - Log-transform skewed numerics (`Age`, `Fee`, `SentimentMagnitude`)  
   - Standard-scale numeric columns, one-hot-encode 12+ categoricals  

3. **Training & Evaluation**  
   - TensorFlow `Dataset` pipelines for mixed modalities  
   - Early stopping and LR reduction callbacks  
   - Test evaluation prints final accuracy and saves `.keras` model  

---

## 📊 Results (Typical)

| Model | Input | Test Accuracy |
|--------|--------|---------------|
| `v1_tabular` | Tabular only | ~62 % |
| `v3_multimodal` | Tabular + Images + Sentiment | **~68 – 70 %** |

Multimodal fusion significantly improved generalization versus tabular-only baselines.

---

## 🧩 Repository Structure

```
├── Data/                        # Raw + processed dataset (ignored in git)
│   ├── train.csv
│   ├── train_sentiment.csv
│   ├── train_images/
│   └── processed/
├── models/                      # Saved .keras weights (ignored in git)
├── figures_for_data_analysis/   # Optional EDA plots
├── model_v1_tabular.py
├── model_v2_multimodal.py
├── model_v3_multimodal.py
├── model_v4_multimodal.py
├── model_v5_multimodal.py
├── data_creator.py
├── 1.data_analysis.py
├── IZVESTAJ_Andrija_Milikic_RN_84_23.pdf
└── .gitignore
```

---

## 🧰 Tech Stack

- **Python 3.12**
- **TensorFlow 2.x / Keras**
- **scikit-learn**, **pandas**, **numpy**
- **Hugging Face Datasets** (for data access)
- **Matplotlib / Seaborn** (for analysis)

---

## 🚀 Usage

```bash
# 1. Prepare data
python data_creator.py

# 2. Train baseline tabular model
python model_v1_tabular.py

# 3. Train multimodal version
python model_v3_multimodal.py

# Models are saved to ./models/
```

---

## 🧾 License

MIT License — free for academic and personal use.  
Models trained on public PetFinder.my dataset.

---

## ✍️ Author

**Andrija Milikic**  
Faculty of Computer Science, RAF – Serbia  
Focus: Multimodal Deep Learning & AI Applications

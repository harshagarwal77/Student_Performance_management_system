---
title: Student Performance Prediction
emoji: 🎓
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.23.0"
app_file: app.py
pinned: false
license: mit
short_description: AI-powered student math score prediction using 8 ML models
---

# 🎓 Student Performance Predictor

Predict student math scores using advanced machine learning — powered by **8 different algorithms** with automatic model selection for maximum accuracy.

## ✨ Features

- **8 ML Models** — Random Forest, XGBoost, CatBoost, Gradient Boosting, AdaBoost, KNN, Decision Tree, Linear Regression
- **Auto Model Selection** — Automatically selects the best performing model using GridSearchCV
- **Beautiful UI** — Dark-themed glassmorphism interface with animations
- **AI Insights** — Personalized recommendations based on predictions
- **Score Comparison** — Visual comparison of predicted math vs reading/writing scores

## 📊 Input Features

| Feature | Description | Values |
|---------|-------------|--------|
| Gender | Student's gender | male, female |
| Race/Ethnicity | Ethnic group | group A-E |
| Parental Education | Highest education of parents | some high school → master's degree |
| Lunch Type | Lunch plan indicator | standard, free/reduced |
| Test Preparation | Completion of prep course | none, completed |
| Reading Score | Reading exam score | 0-100 |
| Writing Score | Writing exam score | 0-100 |

## 🧠 How It Works

1. **Data Preprocessing** — Numerical features are scaled, categorical features are one-hot encoded
2. **Model Training** — 8 models are trained with hyperparameter tuning via GridSearchCV (3-fold CV)
3. **Model Selection** — Best model chosen by R² score
4. **Prediction** — Selected model predicts the math score

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## 📁 Project Structure

```
├── app.py                    # Gradio application
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── notebook/data/stud.csv    # Training dataset
├── artifacts/                # Generated model files
├── requirements.txt
└── README.md
```

## 📝 License

MIT License


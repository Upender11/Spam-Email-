# 📧 Spam Email Detection using Machine Learning

This project uses a **Logistic Regression** model and **TF-IDF Vectorization** to classify emails as **Spam** or **Ham** (not spam). The model is trained on a labeled dataset and achieves high accuracy in detecting spam emails based on their content.

## 🔍 Overview

Spam detection is an essential task in email systems to filter out unwanted or potentially dangerous messages. This project demonstrates how to:

- Load and clean an email dataset
- Preprocess textual data using `TfidfVectorizer`
- Train a `LogisticRegression` model
- Evaluate model performance
- Make predictions on custom email inputs

## 🛠️ Tech Stack

- Python 3.x
- pandas
- scikit-learn
- numpy

## 📁 Dataset

The dataset used is `mail_data.csv`, which contains two columns:
- `Category`: Indicates whether the email is "spam" or "ham"
- `Message`: The content of the email

> Ensure the CSV file is placed in the correct directory:  
`C:\Users\HP\OneDrive\Desktop\JOb\mail_data.csv`

## 🚀 How to Run

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Upender11/Spam-Email-.git
   cd Spam-Email-

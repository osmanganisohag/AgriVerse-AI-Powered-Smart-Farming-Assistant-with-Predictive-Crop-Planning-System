# AgriVerse: AI-Powered Smart Farming Assistant with Predictive Crop Planning System

**AgriVerse** is a dual-function digital platform designed to support farmers with **agricultural knowledge** and **data-driven crop recommendations**. It integrates a **GPT-2 AI assistant** for answering queries and a **Machine Learning crop recommendation system** that predicts suitable crops based on soil and environmental parameters.

---

## Features

- **AI Agricultural Assistant**
  - Fine-tuned GPT-2 model for agricultural Q&A.
  - Provides accurate, context-aware responses to farmer queries.
  - Side-by-side comparison with vanilla GPT-2.

- **Crop Recommendation System**
  - Predicts suitable crops using Random Forest, SVM, and Logistic Regression.
  - Inputs: Nitrogen (N), Phosphorus (P), Potassium (K), pH, rainfall, temperature, humidity.
  - Random Forest achieved **99.32% accuracy**.

- **Web Application**
  - Built with **Flask**.
  - Intuitive interface with navigation between AI Assistant and Crop Recommendation.
  - Simple forms and clear output displays for farmers.

---

## Notebook Analysis Visualizations

**Agricultural QA Dataset**

![Question WordCloud](Images/Most%20Frequent%20Words%20in%20Questions.png)
*WordCloud of most frequent words in questions*

![Answer WordCloud](Images/Most%20Frequent%20Words%20in%20Answers.png)
*WordCloud of most frequent words in answers*

![Question Length Distribution](Images/Distribution%20of%20Question%20Lengths.png)
*Histogram showing question length distribution*

![Answer Length Distribution](Images/Distribution%20of%20Answer%20Lengths.png)
*Histogram showing Answer length distribution*

**Crop Recommendation Dataset**

![Feature Distribution](Images/Feature%20Distributions%20(Histograms%20+%20KDE).png)
*Histograms and KDE of soil and environmental features*

![Class Distribution](Images/Distribution%20or%20Crop%20Type.png)
*Bar chart showing balanced crop categories*

![Confusion Matrix](Images/Random%20Forest%20Confusion%20Matrix.png)
*Confusion matrix for Random Forest crop prediction*

---

## Demo

![Homepage](Images/Homepage.png)
*AgriGPT Homepage*

![AI Assistant](Images/AI%20Assistant.png)
*AI Assistant generating responses*

![Crop Recommendation](Images/Crop%20Recommend.png)
*Crop Recommendation page predicting suitable crops*

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/osmanganisohag/AgriGPT-Enhancing-Agriculture-with-Fine-Tuned-Language-Models-and-Crop-Prediction.git](https://github.com/mdjisan1/AgriGPT-Enhancing-Agriculture-with-Fine-Tuned-Language-Models-and-Crop-Prediction.git)
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd AgriGPT
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask application:**
    ```bash
    python app.py
    ```

5.  Open your browser at `http://127.0.0.1:5000`.

---

## Datasets

-   **Agricultural Q&A Dataset:** [KisanVaani (Hugging Face)](https://huggingface.co/datasets/kisanvaani/kisan-vaani-QA), 18,403 preprocessed Q&A pairs.
-   **Crop Recommendation Dataset:** [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset), 2200 samples, 22 crops.

---

## Model Details

-   **AI Assistant:** GPT-2 fine-tuned using Hugging Face Transformers and PyTorch.
-   **Crop Recommendation:** Random Forest selected as best performer; SVM and Logistic Regression also trained for comparison.

---

## Future Work

-   Multilingual GPT models for regional languages (Bangla, Hindi).
-   Integration with IoT sensors for real-time soil and weather data.
-   Mobile application support for offline usage.
-   Federated learning for privacy-preserving Q&A.

---

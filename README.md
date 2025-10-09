# Employee-Turnover-Prediction
This project uses AI to combat costly employee turnover. By analyzing HR data, it predicts at-risk employees, assesses their efficiency, and estimates the financial impact of their departure. This enables HR teams to shift from reactive measures to proactive, data-driven retention strategies, fostering a more stable workforce.

# Predictive Analytics for Employee Retention and Turnover Forecasting

An AI-powered system built with Django to predict employee turnover, assess employee efficiency, and estimate the financial impact of attrition, enabling proactive HR strategies.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white)]()
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)]()
[![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)]()
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)]()

---

## 📖 Table of Contents
* [About The Project](#about-the-project)
* [Key Features](#key-features)
* [Project Flow](#project-flow)
* [Tech Stack](#tech-stack)
* [Getting Started](#getting-started)
* [Project Team](#project-team)
* [Acknowledgments](#acknowledgments)

---

## 🎯 About The Project

[cite_start]Employee turnover is a critical challenge that impacts the productivity, stability, and finances of organizations worldwide[cite: 37]. [cite_start]Traditional HR methods are often reactive, failing to provide the timely insights needed to retain valuable employees[cite: 38]. [cite_start]This leads to high costs associated with recruitment, training, and lost productivity[cite: 27, 39].

[cite_start]This project introduces an AI-powered system to transform how organizations manage attrition risks[cite: 40]. [cite_start]By analyzing HR data and performance metrics, it predicts the likelihood of an employee leaving[cite: 41]. [cite_start]The system also calculates a unique **efficiency score** to quantify an employee's contribution and estimates the **financial cost** of their potential departure[cite: 42, 43]. [cite_start]These insights are delivered through a smart analytical dashboard, empowering HR managers with the tools for data-driven, proactive decision-making to foster a more engaged and stable workforce[cite: 33, 44].

---

## ✨ Key Features

* [cite_start]**🤖 AI-Based Turnover Prediction**: Utilizes machine learning models, including **Logistic Regression**, **Random Forest**, and **HistGradientBoosting**, to accurately identify employees at high risk of leaving[cite: 58, 156].
* [cite_start]**📈 Employee Efficiency Scoring**: Implements a custom scoring mechanism to assess employee contribution based on performance evaluations, project involvement, monthly hours, and tenure[cite: 59, 157].
* [cite_start]**💰 Turnover Cost Estimation**: Integrates a module to calculate the financial impact of losing an employee, helping managers prioritize retention efforts for high-value individuals[cite: 60, 158].
* [cite_start]**📊 Interactive HR Dashboard**: Features a user-friendly interface that visualizes turnover risks, efficiency levels, and cost analysis across different departments[cite: 118, 159].
* [cite_start]**🚀 Proactive HR Interventions**: Provides predictive insights that enable HR teams to implement timely engagement strategies, promotions, or training programs for at-risk employees before they decide to leave[cite: 61, 89].

---

## 🌊 Project Flow

[cite_start]The system follows a systematic methodology, from data ingestion to actionable HR insights[cite: 64]. [cite_start]The core process includes data preprocessing, efficiency scoring, and predictive modeling to generate results that inform retention strategies[cite: 65].



1.  [cite_start]**Data Collection**: The system gathers comprehensive employee data from HR records, including demographics, performance metrics, and employment history[cite: 93].
2.  **Data Preprocessing**: Raw data is cleaned and prepared for modeling. [cite_start]This involves handling missing values through imputation, encoding categorical features, and normalizing numerical data[cite: 99].
3.  [cite_start]**Efficiency Scoring & ML Modeling**: A weighted efficiency score is calculated for each employee[cite: 103]. [cite_start]Concurrently, machine learning models (Logistic Regression, Random Forest, HistGradientBoosting) are trained to predict turnover risk[cite: 106, 107].
4.  [cite_start]**Results & Visualization**: The outputs—predicted turnover risk, efficiency score, and estimated cost—are presented on a dashboard, allowing HR managers to rank employees and identify high-priority cases for intervention[cite: 118, 120].

---

## 🛠️ Tech Stack

This project leverages a modern tech stack to deliver a robust and scalable solution.

* **Backend Framework**: **Django**
* **Frontend**: **HTML**, **CSS**
* [cite_start]**AI & Machine Learning**: **Python**, Scikit-learn, Pandas, NumPy, Matplotlib [cite: 183]
* [cite_start]**Database**: MySQL / PostgreSQL [cite: 182]
* [cite_start]**Development Tools**: Visual Studio Code, Git & GitHub [cite: 184]

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.x
* Pip
* Git

### Installation

1.  **Clone the repo**
    ```sh
    git clone [https://github.com/your_username/your_project_repository.git](https://github.com/your_username/your_project_repository.git)
    ```
2.  **Navigate to the project directory**
    ```sh
    cd Project_Final
    ```
3.  **Install Python dependencies**
    ```sh
    pip install -r requirements.txt
    ```
4.  **Run Python code for Employee efficiency, Turnover and Turnover cost**
    ```sh
    python ER5.py
    ```
5.  **Run the Visulization code**
    ```sh
    python Visualization.py
    ```
    Check the new CSV files and Visualization folder Created in your Project Folder 

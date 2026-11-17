# 💰 FinSight AI – AI-Powered Financial Expense Analyzer

An intelligent personal finance management application that helps users track expenses, analyze spending patterns, predict future expenses, and gain financial insights using Machine Learning and Generative AI.

## 🚀 Project Overview

**FinSight AI** is a smart financial analytics application designed to simplify personal expense management. It transforms transaction data into meaningful financial insights through interactive dashboards, machine learning-based expense categorization, predictive analytics, and an AI-powered financial assistant.

The application enables users to understand their financial habits, monitor savings, identify spending patterns, and make data-driven financial decisions.
## 🎥 Project Demo

<a href="PASTE_YOUTUBE_VIDEO_URL_HERE">
  <img src="" alt="FinSight AI Demo" width="700">
</a>

## ✨ Key Features

### 📊 1. Interactive Financial Dashboard
- Track total income, expenses, and savings.
- Monitor monthly financial performance.
- Visualize income vs. expenses.
- Analyze savings rate.
- View category-wise spending breakdown.

### 💳 2. Transaction Management
- Upload transaction data using CSV files.
- View and manage financial transactions.
- Search and filter transactions.
- Organize transactions by category and payment method.

### 📈 3. Expense Analysis
- Analyze category-wise expenses.
- Identify top spending categories.
- Track income and expense patterns.
- Visualize payment method distribution.
- Generate meaningful financial insights.

### 🤖 4. Machine Learning-Based Expense Categorization
- Automatically categorize transactions.
- Uses TF-IDF for text feature extraction.
- Implements a Random Forest classification model.
- Reduces manual transaction categorization.

### 🔮 5. Expense Prediction
- Forecast future monthly expenses.
- Analyze historical spending patterns.
- Apply predictive analytics to estimate upcoming expenses.
- Evaluate prediction performance using MAE, RMSE, and R².

### 💬 6. AI Financial Assistant
- AI-powered conversational financial assistant.
- Ask questions about financial data.
- Receive spending insights and financial guidance.
- Integrated with the Groq API.

### ⚙️ 7. Settings
- Configure AI provider settings.
- Manage application preferences.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / UI | Streamlit |
| Data Processing | Pandas, NumPy |
| Data Visualization | Plotly |
| Machine Learning | Scikit-learn |
| NLP | TF-IDF |
| ML Algorithms | Random Forest, Regression |
| Generative AI | Groq API |
| Database | SQLite |
| Development Tools | VS Code, Git, GitHub |

## 📁 Project Structure

```text
FinSight-AI/
│
├── assets/
│
├── data/
│
├── database/
│   ├── __init__.py
│   ├── db_connection.py
│   └── db_operations.py
│
├── models/
│   └── train_model.py
│
├── modules/
│   ├── ai_assistant.py
│   ├── data_processing.py
│   ├── expense_categorization.py
│   ├── expense_prediction.py
│   ├── financial_analytics.py
│   └── report_generation.py
│
├── pages/
│   ├── assistant.py
│   ├── categorization.py
│   ├── dashboard.py
│   ├── expense_analysis.py
│   ├── prediction.py
│   ├── reports.py
│   ├── settings.py
│   └── transactions.py
│
├── tests/
│
├── .env.example
├── .gitignore
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Setup

Follow these steps to run FinSight AI locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Mamta-18/FinSight-AI-AI-Powered-Financial-Expense-Analyzer.git
```

### 2. Navigate to the Project Directory

```bash
cd FinSight-AI-AI-Powered-Financial-Expense-Analyzer
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project root directory and add your Groq API key using the environment variable name expected by your application.

Example:

```env
GROQ_API_KEY=your_groq_api_key
```

**Important:** Never upload your actual API key or `.env` file to GitHub.

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📂 Dataset

FinSight AI supports CSV-based transaction uploads.

The transaction dataset can contain fields such as:

- Transaction ID
- Date
- Description
- Amount
- Transaction Type
- Category
- Payment Method

Users can upload their own transaction data to explore financial patterns and generate insights.

## 🧪 Testing

The project includes test modules for different application components, including:

- AI Assistant
- Expense Categorization
- Data Processing
- Financial Analytics
- Expense Prediction

Run the tests using:

```bash
pytest
```

## 🎯 Project Objectives

- Simplify personal financial management.
- Automate transaction categorization using Machine Learning.
- Identify spending patterns through data analytics.
- Predict future expenses using historical transaction data.
- Provide conversational financial insights using Generative AI.
- Support data-driven financial decision-making.

## 🔮 Future Enhancements

- Integration with banking APIs for automated transaction imports.
- Advanced financial forecasting models.
- Personalized savings recommendations.
- Budget planning and spending alerts.
- PDF financial report generation.
- Cloud deployment and user authentication.

## 👩‍💻 Author

**Mamta Rathore**

Computer Science Graduate | PGDM – Data Science & Applied AI

Interested in Data Analytics, Machine Learning, and Generative AI.

🔗 **LinkedIn:** [Mamta Rathore](https://www.linkedin.com/in/mamta-rathore-063577316/)

🔗 **GitHub:** [Mamta-18](https://github.com/Mamta-18)

---

⭐ If you find this project interesting, consider giving the repository a star!

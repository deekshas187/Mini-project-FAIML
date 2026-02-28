# Stock Price Trend Prediction using MongoDB and Machine Learning
## Project Overview
This project focuses on analyzing stock market data and predicting stock price trends using Machine Learning. The system uses historical stock data and applies a Random Forest Classifier to predict whether the stock trend is Up or Down.

The project also integrates MongoDB for storing and retrieving stock data.

---

## Features
- Stock Closing Price Trend Visualization
- Volume Analysis Graph
- Correlation Heatmap of Features
- MongoDB Database Integration
- Random Forest Machine Learning Model
- Model Accuracy Evaluation
- Example Trend Prediction

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- MongoDB
- PyMongo

---

## Dataset Information
The dataset contains the following columns:

- Date
- Open
- High
- Low
- Close
- Volume
- Trend (Target variable: Up / Down)

---

## How to Run the Project

### Step 1: Install Python
Make sure Python is installed on your system.

### Step 2: Install Required Libraries
```bash
pip install pandas pymongo matplotlib seaborn scikit-learn
```

### Step 3: Install and Run MongoDB
Ensure MongoDB is installed and running on:
```
mongodb://localhost:27017/
```

### Step 4: Run the Python File
```bash
python stock_project.py
```

---

## Model Performance

- Model Used: Random Forest Classifier
- Accuracy Achieved: 77.5%
- Evaluation Metrics:
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## Sample Prediction

Example Input:

Open: 150
High: 152
Low: 149
Close: 151
Volume: 1200000

Predicted Output:
Trend: Up

## Project Outcome
- Successfully integrated Machine Learning with a database.
- Achieved reliable trend prediction accuracy.
- Visualized stock patterns effectively.
- Demonstrated real-world data processing workflow.

---

## Author
Deeksha S  
4th CSE A  
Mini Project - FAIML

---

## Future Improvements
- Deploy as a Web Application
- Add Real-Time Stock Data API
- Improve Model Accuracy using advanced algorithms
- Add Hyperparameter Tuning

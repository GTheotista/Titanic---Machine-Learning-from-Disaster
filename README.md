# Titanic Survival Prediction

## **Project Overview**
The Titanic disaster is one of the most infamous maritime tragedies in history. By analyzing passenger data, this project aims to build a machine learning model that predicts whether a passenger survived based on available features.

## **Dataset**
The dataset used in this project is from Kaggle's [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data?select=train.csv). It contains information about passengers, including ticket class, age, fare, and embarkation port.

## **Tools Used**
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, LightGBM
- **Visualization:** Matplotlib, Seaborn

## **Project Workflow**
### **1. Data Understanding**
We select relevant features for model training, including:
- **Pclass** (Passenger class)
- **Age** (Age of the passenger)
- **Sex** (Gender of the passenger)
- **SibSp** (Number of siblings/spouses aboard)
- **Parch** (Number of parents/children aboard)
- **Fare** (Fare paid for the ticket)
- **Embarked** (Port of embarkation)

### **2. Data Preprocessing**
- **Handling Missing Values:**
  - `Age` is imputed with the median value.
  - `Embarked` is imputed with the mode (most frequent value).
- **Encoding Categorical Features:**
  - `Embarked` and `Sex` are converted into numerical representations using one-hot encoding.
- **Feature Scaling:**
  - StandardScaler is applied to numerical features to normalize the data.

### **3. Model Training & Evaluation**
Four machine learning models were trained and evaluated based on accuracy:
- **Logistic Regression**
- **Decision Tree Classifier**
- **LightGBM Classifier**
- **Gradient Boosting Classifier**

### **4. Results**
The accuracy scores of each model are as follows:
- **Logistic Regression:** 77.13%
- **Decision Tree:** 74.44%
- **LightGBM:** 77.58%
- **Gradient Boosting:** 79.82%

### **5. Conclusion**
- Gradient Boosting achieved the highest accuracy at **79.82%**, making it the best-performing model.
- Feature importance analysis could be conducted to further understand which features have the most impact on survival.
- Future improvements could include hyperparameter tuning, feature engineering, and handling class imbalance.
---
**Author:** [Giovanny Theotista]  


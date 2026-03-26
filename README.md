# CodeAlpha_Tasks
# TASK 3: Car Price Prediction with Machine Learning

This notebook presents a complete machine learning workflow for predicting the selling price of used cars based on historical and technical vehicle attributes. The goal is to build a reliable regression model that can estimate car prices by learning patterns from data, similar to the pricing engines used by dealerships, online marketplaces, and automotive analytics platforms.

Objectives
1. Explore and Understand the Dataset
We begin by examining the available features, including vehicle brand, model year, mileage, fuel type, transmission, and original showroom price. This helps us understand how different characteristics influence resale value.

2. Perform Data Preprocessing
The dataset is cleaned and prepared by handling missing values, correcting data types, and transforming raw fields into usable formats. We also encode categorical variables and create new features, such as the age of the car, to improve predictive power.

3. Conduct Feature Engineering
We enhance the dataset by extracting meaningful patterns (e.g., vehicle age, brand categories) and analyzing distributions and outliers. This step helps refine the dataset so that models can learn more effectively.

4. Train Regression Models
We develop and compare several machine learning algorithms, including both linear and tree‑based models. Each model is trained to predict car prices using the engineered features.

5. Evaluate Model Performance
Using metrics such as R^2, MAE, and RMSE, we assess model accuracy and generalization. Visualization techniques are used to interpret model behavior and identify the strongest predictors.

6. Demonstrate Real‑World Application
The project highlights how machine learning can support price estimation in real automotive contexts, helping dealerships, buyers, and sellers make more informed decisions.

Libraries
Python for data processing and modeling
Pandas for data manipulation
NumPy for numerical operations
Matplotlib & Seaborn for visualization
Scikit‑learn for preprocessing, modeling, and evaluation

# Task 4: Sales Prediction using Python
**Overview**
This notebook performs an in-depth analysis of advertising impact on sales using the Advertising.csv dataset. It builds and evaluates various regression models, calculates sales elasticities for different advertising channels (TV, Radio, Newspaper), runs sales uplift scenarios, and proposes an optimal budget allocation strategy.

# Dataset
The analysis utilizes the Advertising.csv dataset, which contains information on advertising spend for TV, Radio, and Newspaper, and the corresponding Sales.

**Key Analyses & Methodologies**
1. Data Loading & Exploration: Initial data loading, inspection of data types, dimensions, and descriptive statistics.
2. Data Preprocessing: Handling of an Unnamed column and outlier detection and removal to ensure robust model training.

3. Exploratory Data Analysis (EDA):
 
 - - Box plots for outlier analysis.
 - * Pair plots to visualize relationships between advertising spend and sales.
 + * Heatmap to show correlations between all variables.

5. Model Building & Evaluation:
* Linear Regression: A baseline linear model was trained to understand the direct relationship between advertising spend and sales.
* Ridge Regression: A regularized linear model was trained, with hyperparameter tuning using GridSearchCV and cross-validation.
* Random Forest Regressor: A powerful ensemble model was used to capture non-linear relationships.
* Models were evaluated using RMSE, MAE, and MAPE metrics on a test set.

5. Statistical Summary (OLS): An OLS (Ordinary Least Squares) model from statsmodels was used to provide a detailed statistical summary, including p-values and confidence intervals for coefficients, revealing the significance of each advertising channel.
6. Elasticity Calculation: Log-log linear regression was employed to calculate the sales elasticity for each advertising channel, quantifying the percentage change in sales for a 1% change in advertising spend.
7. Scenario Simulation: A simulator was developed to project sales uplift based on a hypothetical 20% increase in spend for each advertising channel.
8. Budget Optimization: Based on the calculated elasticities and using a Cobb-Douglas shares approach, an optimal budget allocation across TV, Radio, and Newspaper was proposed to maximize sales impact.
9. Visualization: Various plots were generated to visualize data distributions, correlations, model results, sales uplifts from scenarios, and the proposed optimal budget allocation against current spend.

# Key Findings
* **Strong Predictors:** TV and Radio advertising were found to be highly significant predictors of sales, with TV having the strongest impact.
* **Weak Predictor:** Newspaper advertising showed a very weak or statistically insignificant impact on sales.
* **Elasticities:**
* TV had the highest elasticity (~0.36), meaning a 1% increase in TV spend leads to ~0.36% sales increase.
* Radio had a moderate elasticity (~0.20).
* Newspaper had a very low elasticity (~0.017), indicating minimal sales response to spend changes.

* **Scenario Uplifts (20% Spend Increase):**
* TV: ~9.56% sales uplift.
* Radio: ~5.98% sales uplift.
* Newspaper: ~0.22% sales uplift.
* **Optimal Budget Allocation:** The analysis suggests reallocating budget significantly towards Radio advertising, slightly reducing TV spend, and drastically decreasing Newspaper advertising spend to achieve a more efficient and impactful advertising strategy.

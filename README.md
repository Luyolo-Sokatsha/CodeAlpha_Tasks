# Car Price Prediction with Machine Learning - README

## Overview
This notebook presents a complete machine learning workflow for predicting the selling price of used cars based on historical and technical vehicle attributes. The goal is to build a reliable regression model that can estimate car prices by learning patterns from data, similar to the pricing engines used by dealerships, online marketplaces, and automotive analytics platforms.

## Dataset
The dataset `car data.csv` contains various features related to used cars, including:
- `Car_Name`: Name of the car model
- `Year`: Manufacturing year of the car
- `Selling_Price`: The price at which the car was sold (target variable)
- `Present_Price`: The current showroom price of the car
- `Driven_kms`: Total kilometers driven by the car
- `Fuel_Type`: Type of fuel used (Petrol, Diesel, CNG)
- `Selling_type`: How the car was sold (Dealer, Individual)
- `Transmission`: Type of transmission (Manual, Automatic)
- `Owner`: Number of previous owners

## Workflow

### 1. Importing Libraries
Necessary libraries such as `numpy`, `pandas`, `matplotlib.pyplot`, and `seaborn` were imported for data manipulation, analysis, and visualization.

### 2. Data Loading and Initial Exploration
- The `car data.csv` dataset was loaded into a pandas DataFrame.
- Initial data inspection was performed using `df.head()`, `df.shape`, `df.info()`, and `df.describe()` to understand its structure, dimensions, data types, and statistical summary.

### 3. Data Cleaning
- **Missing Values:** Checked for and confirmed no missing values (`df.isnull().sum()`).
- **Duplicate Values:** Identified and removed duplicate rows (`df.duplicated().sum()`, `df.drop_duplicates()`).

### 4. Data Visualization
- Explored the distribution of categorical features (`Fuel_Type`, `Selling_type`, `Transmission`) using `value_counts()`.
- Visualized distributions of numerical features (`Selling_Price`, `Present_Price`, `Driven_kms`) using histograms and KDE plots, often segmented by `Transmission` type.
- Analyzed price distribution using `sns.histplot(df['Selling_Price'], kde=True)`.
- Investigated relationships between `Selling_Price` and `Year`, `Present_Price`, `Driven_kms` using scatter plots.
- Created a pairplot for `Selling_Price`, `Present_Price`, `Driven_kms` to visualize pairwise relationships.
- Used boxplots to understand `Selling_Price` distribution across different `Fuel_Type` and `Transmission` types.
- Identified top car models by average selling price.
- Count plots were used to visualize the distribution of `Fuel_Type`, `Selling_type`, `Transmission`, `Owner`, and the top 10 `Car_Name`.

### 5. Categorical Feature Encoding
- Categorical columns (`Fuel_Type`, `Selling_type`, `Transmission`) were encoded into numerical representations using `df.replace()`:
    - `Fuel_Type`: Petrol (0), Diesel (1), CNG (2)
    - `Selling_type`: Dealer (0), Individual (1)
    - `Transmission`: Manual (0), Automatic (1)

### 6. Feature Correlation
- A heatmap was generated using `sns.heatmap(df.corr(numeric_only=True))` to visualize the correlation matrix between numerical features.

### 7. Data Preprocessing
- **Normalization and Scaling:** Numerical features (`Selling_Price`, `Present_Price`, `Driven_kms`) were scaled using `StandardScaler` to have a mean of 0 and standard deviation of 1.
- **Feature Engineering:** A new feature `Car_Age` was created by subtracting the `Year` from the current year, and the original `Year` column was dropped.
- **Outlier Handling:** Outliers in `Selling_Price`, `Present_Price`, and `Driven_kms` were removed using the Interquartile Range (IQR) method.
- A pairplot of numerical features was re-generated after outlier removal to confirm the effect.

### 8. Model Training
- **Data Splitting:** The dataset was split into features (`X`) and target (`y`, which is `Selling_Price`). The `Car_Name` column was dropped from `X` as it's not directly used in this model.
- The data was further split into training (80%) and testing (20%) sets using `train_test_split`.
- **Decision Tree Regressor:**
    - A `DecisionTreeRegressor` model was trained on the `X_train` and `y_train` data.
    - Predictions were made on the `X_test` set.
    - Model performance was evaluated using Mean Squared Error (MSE) and R-squared (R^2).
- **Random Forest Regressor:**
    - A `RandomForestRegressor` model was trained.
    - Predictions were made on the `X_test` set.
    - Model performance was evaluated using Mean Squared Error (MSE) and R-squared (R^2).
    - Feature importances were plotted to identify the most influential features for prediction.
    - A regression plot of actual vs. predicted `Selling_Price` was created to visualize the model's performance.

### 9. Conclusion
The `RandomForestRegressor` model yielded better prediction results (lower MSE, higher R^2) compared to the `DecisionTreeRegressor`, indicating its suitability for car price prediction in this scenario.


## README: Sales Prediction using Python

This notebook provides a comprehensive analysis of the impact of advertising spend across different channels (TV, Radio, Newspaper) on Sales, using the `Advertising.csv` dataset. It covers data exploration, model building, elasticity estimation, scenario simulation, and optimal budget allocation.

### 1. Overview
This notebook aims to:
*   Build and evaluate various regression models to predict sales.
*   Estimate the sales elasticity of different advertising channels.
*   Simulate the impact of changes in advertising spend on sales.
*   Propose an optimal budget allocation strategy based on advertising elasticities.

### 2. Dataset
The analysis is based on the `Advertising.csv` dataset, which contains information on advertising budgets (in thousands of dollars) for TV, radio, and newspaper media, and corresponding sales figures (in thousands of units).

**Columns:**
*   `TV`: Advertising spend on TV.
*   `Radio`: Advertising spend on Radio.
*   `Newspaper`: Advertising spend on Newspaper.
*   `Sales`: Sales of a product.

### 3. Analysis & Modeling

#### a. Data Preparation
The data is split into training and testing sets (80/20 ratio) for model development and evaluation.

#### b. Regression Models
Three different regression models are implemented and evaluated:
*   **Linear Regression**: A baseline model to understand linear relationships.
*   **Ridge Regression**: A regularized linear model to prevent overfitting, with `alpha` tuned using GridSearchCV.
*   **Random Forest Regressor**: A robust ensemble model capable of capturing non-linear relationships.

#### c. Model Evaluation
Models are evaluated using:
*   **RMSE** (Root Mean Squared Error)
*   **MAE** (Mean Absolute Error)
*   **MAPE** (Mean Absolute Percentage Error)

#### d. Elasticity Estimation
Sales elasticities for each advertising channel are estimated using a log-log linear regression model. This provides insights into the percentage change in sales for a 1% change in advertising spend.

#### e. Scenario Simulation
The notebook includes a function to simulate the impact of percentage changes in advertising spend on sales, based on the Linear Regression model. This helps in understanding the potential uplift in sales from targeted budget adjustments.

#### f. Budget Optimization
An optimal budget allocation is proposed based on the calculated elasticities, using a Cobb-Douglas shares approach. This method aims to maximize sales by distributing the total advertising budget proportionally to the elasticities of each channel.

### 4. Key Findings
*   **Random Forest Regressor** generally demonstrates the best predictive performance (lowest RMSE, MAE, and MAPE) among the tested models.
*   **TV** advertising consistently shows the highest impact on sales, followed by **Radio**. **Newspaper** advertising has a significantly lower impact.
*   The **elasticity values** confirm that TV and Radio have a much stronger influence on sales compared to Newspaper.
*   The **optimal budget allocation** suggests a higher proportion of spending on TV and Radio, and a considerably lower allocation for Newspaper advertising, aligning with their respective elasticities.

### 5. Visualizations
The notebook includes several visualizations to illustrate the findings:
*   **Correlation Heatmap**: Visualizes the correlations between advertising spend and sales.
*   **Budget Comparison Bar Chart**: Compares current average advertising spend with the proposed optimal budget allocation.
*   **Sales Uplift Bar Chart**: Displays the percentage sales uplift from a 20% increase in spend for each channel.
*   **Sales Elasticity Bar Chart**: Illustrates the estimated sales elasticity for each advertising channel.

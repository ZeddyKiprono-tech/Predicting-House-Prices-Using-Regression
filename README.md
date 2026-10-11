# Predicting House Prices Using Linear Regression

## Project Overview

This project uses supervised machine learning to estimate median house values from housing and neighborhood characteristics. 
It is designed around a real estate analytics problem, understanding how selected property and neighborhood features relate 
to house prices and building a model that can estimate values for previously unseen observations.

The analysis follows a basic machine learning workflow, inspecting and validating the data, exploring relationships between variables,
splitting the data into training and testing sets, training a linear regression model, evaluating its predictions, and examining model
coefficients.

## Business Problem

A real estate analyst wants to understand how neighborhood characteristics relate to house prices and build a model that estimates
property values from available housing data.

**Objective:** 
Develop a regression model that uses housing and neighborhood features to predict median owner-occupied home values.

-   **Machine learning task:** Supervised learning --- regression
-   **Target variable:** `MEDV`
-   **Target meaning:** Median owner-occupied home value, recorded in thousands of US dollars in the dataset

## Dataset
The dataset used is 'boston.csv' containing **506 observations and 14 columns**: 13 predictor variables and the target variable `MEDV`. 
From the data inspection no missing values were found, and the columns were numeric.

### Variables used
The model uses all columns except `MEDV` as predictors:
  --------------------------------------------------------------------------------------------------------
  Variable                            Description in the analysis
  ----------------------------------- ---------------------------------------------------------------------
  `CRIM`                              Per-capita crime rate by town
  `ZN`                                Proportion of residential land zoned for lots over 25,000 sq. ft.
  `INDUS`                             Proportion of non-retail business acres per town
  `CHAS`                              Charles River dummy variable
  `NOX`                               Nitric oxide concentration
  `RM`                                Average number of rooms per dwelling
  `AGE`                               Proportion of owner-occupied units built before 1940
  `DIS`                               Weighted distance to employment centres
  `RAD`                               Index of accessibility to radial highways
  `TAX`                               Property-tax rate
  `PTRATIO`                           Pupil-teacher ratio by town
  `B`                                 Dataset-provided demographic transformation variable
  `LSTAT`                             Dataset-provided lower-status population percentage variable
  `MEDV`                              Median owner-occupied home value; the target to predict
  -----------------------------------------------------------------------------------------------------------

## Methodology

### 1. Data Inspection and Quality Checks
-   Loaded the dataset into a Pandas DataFrame.
-   Previewed records to understand the data structure.
-   Checked column data types and dataset dimensions.
-   Generated descriptive statistics to review distributions and ranges.
-   Checked for missing values

### 2. Exploratory Data Analysis

A scatter plot was used to explore the relationship between `RM` (average number of rooms) and `MEDV` (median house value).

#### Findings : 
Two variables have a positive a relationship that is homes in areas with a higher average number of rooms tend to have higher median 
values in this dataset. 

### 3. Feature and Target Selection

-   **Features (`X`):** All columns except `MEDV`.
-   **Target (`y`):** `MEDV`.

### 4. Train-Test Split
The data was split into: 
- **80% training data** to fit the model.
- **20% testing data** to evaluate it on held-out observations.

The split uses `random_state=42` to make the results reproducible.

### 5. Model Training and Prediction

A `LinearRegression` model from Scikit-learn was fitted on the training data and used to predict `MEDV` values for the test set.

### 6. Model Evaluation
The analysis calculates Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.

## Results

The following evaluation results on the test set:
  ---------------------------------------------------------------------------------------
  Metric                                  Result Interpretation
  --------------------- ---------------------------- ------------------------------------
  Mean Absolute Error (MAE)               3.19 Predictions differ from actual values by
                                          about 3.19 target units on average in absolute terms.    

  Root Mean Squared Error (RMSE)          4.93 Larger prediction errors receive more
                                          weight than they do under MAE.

  R-squared (R²)                          0.67 The model explains approximately 67% of
                                          the variation in the test-set target values.   
                                                     
  -------------------------------------------------------------------------------------------

Because `MEDV` is expressed in thousands of US dollars in the dataset, an MAE of 3.19 corresponds to approximately **\$3,190** in the dataset's
original units. These metrics describe performance on this particular test split; they do not guarantee the same performance on other data.

### Actual vs. Predicted Values
A dashed diagonal line represents perfect predictions. Points closer to the line indicate smaller prediction errors, while points farther away 
indicate larger errors.

### Feature Coefficients

The model coefficients show the direction and size of each feature's linear relationship with the predicted target while the other included
features are held constant in the model.
Selected coefficients from the notebook include:
-   `RM`: **+4.44**
-   `CHAS`: **+2.78**
-   `NOX`: **−17.20**
-   `DIS`: **−1.45**
-   `PTRATIO`: **−0.92**
-   `LSTAT`: **−0.51**

The coefficient magnitudes should be interpreted cautiously because the predictors use different scales. Coefficients indicate associations
learned by the model; they do not establish causation or, on their own, prove which variables are most important.

## Tools and Libraries

-   **Python** --- programming language used for the analysis
-   **Pandas** --- data loading, inspection, and tabular data handling
-   **NumPy** --- numerical calculations, including RMSE
-   **Matplotlib** --- plotting
-   **Seaborn** --- statistical visualization
-   **Scikit-learn** --- train-test splitting, linear regression, and model evaluation
-   **Jupyter Notebook / Google Colab** --- interactive analysis environment

## Business Considerations and Limitations

-   The model is a baseline linear regression model and may not capture nonlinear relationships or interactions between features.
-   The evaluation uses one train-test split. Cross-validation and comparison with alternative models could provide a more robust
    assessment.
-   The notebook does not include a residual analysis, hyperparameter/model comparison, or external validation.
-   Historical housing data may not represent current market conditions.
    The model should not be used for real-world pricing decisions without checking data provenance, relevance, fairness, and current
    market performance.
-   The dataset includes demographic-related variables. Their use in property-related decision-making requires particular care, since
    historical patterns may reflect structural inequities.

## Conclusion

This project demonstrates a foundational machine learning workflow for a regression problem: inspecting data, exploring relationships, selecting
predictors and a target, splitting data, training a linear regression model, generating predictions, and evaluating performance.

The model achieved an R² of 0.67 on the notebook's test split, with an MAE of 3.19 and an RMSE of 4.93 in the dataset's target units. The
results provide a useful baseline for learning and further experimentation, rather than a validated tool for making current
property valuations.

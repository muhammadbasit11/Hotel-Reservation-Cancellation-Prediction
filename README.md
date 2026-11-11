# Hotel Reservation Cancellation Prediction

## Project Overview

This project focuses on predicting whether a hotel reservation will be canceled or not using Machine Learning classification algorithms.

The project uses hotel booking information such as the number of guests, lead time, room type, meal plan, previous booking history, average room price, and special requests.

The main focus is on model evaluation and fine-tuning through K-Fold Cross Validation, Stratified K-Fold Cross Validation, Grid Search CV, and Randomized Search CV.

## Problem Statement

Given historical hotel reservation data, the objective is to build classification models that predict whether a booking will be canceled or not.

The target variable is `booking_status`:

- `Not_Canceled`: Booking was not canceled.
- `Canceled`: Booking was canceled.

## Dataset Information

- **Dataset:** Hotel Reservations Classification Dataset
- **Total records:** 36,275
- **Original columns:** 19
- **Target variable:** `booking_status`

**Dataset source:** [Hotel Reservations Classification Dataset – Kaggle](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)

### Features

| Feature | Description |
|---|---|
| no_of_adults | Number of adults |
| no_of_children | Number of children |
| no_of_weekend_nights | Weekend nights booked |
| no_of_week_nights | Week nights booked |
| type_of_meal_plan | Meal plan selected |
| required_car_parking_space | Whether parking is required |
| room_type_reserved | Reserved room type |
| lead_time | Days between booking and arrival |
| market_segment_type | Booking market segment |
| repeated_guest | Whether the customer is a repeated guest |
| no_of_previous_cancellations | Previous canceled bookings |
| no_of_previous_bookings_not_canceled | Previous successful bookings |
| avg_price_per_room | Average room price per day |
| no_of_special_requests | Number of special requests |
| booking_status | Cancellation status (target) |

The notebook removes `booking_id`, `arrival_year`, `arrival_month`, and `arrival_date` during preprocessing.

## Project Workflow

### 1. Data Preprocessing and Exploratory Data Analysis

- Loaded the dataset using Pandas.
- Checked the dataset dimensions and inspected the first few records.
- Removed unnecessary booking and arrival date columns.
- Visualized booking status using a count plot.
- Compared lead time for canceled and non-canceled bookings using box plots.
- Compared average room prices using box plots.
- Applied one-hot encoding to categorical features:
  - `type_of_meal_plan`
  - `room_type_reserved`
  - `market_segment_type`

### 2. Feature Scaling

Applied `StandardScaler` to:

- `lead_time`
- `avg_price_per_room`

Separated the independent features (`X`) and target (`y`).

### 3. K-Fold Cross Validation

Implemented 7-fold K-Fold Cross Validation with shuffling and `random_state=42`.

Evaluated the following classification algorithms:

- Gaussian Naive Bayes
- Logistic Regression
- Decision Tree Classifier

**Results:**

| Algorithm | Mean Accuracy |
|---|---:|
| Gaussian Naive Bayes | 40.86% |
| Logistic Regression | 80.05% |
| Decision Tree | 85.70% |

### 4. Stratified K-Fold Cross Validation

Implemented 7-fold Stratified K-Fold Cross Validation to maintain the class distribution across folds.

Evaluated the same three classification algorithms.

**Results:**

| Algorithm | Mean Accuracy |
|---|---:|
| Gaussian Naive Bayes | 40.86% |
| Logistic Regression | 79.96% |
| Decision Tree | 85.84% |

### 5. Hyperparameter Tuning Using GridSearchCV

Used GridSearchCV to explore hyperparameter combinations for Gaussian Naive Bayes and Random Forest.

**Gaussian Naive Bayes parameters:**
- `var_smoothing`: [1e-9, 1e-8, 1e-7]

**Random Forest parameters:**
- `n_estimators`: [10, 50, 100]
- `max_depth`: [5, 10, 15]

**Results:**

| Model | Best CV Score | Best Parameters |
|---|---:|---|
| Gaussian Naive Bayes | 40.95% | `var_smoothing=1e-7` |
| Random Forest | 87.71% | `n_estimators=100`, `max_depth=15` |

An additional GridSearchCV experiment was performed with a Decision Tree Classifier using different `max_depth` values.

### 6. Hyperparameter Tuning Using RandomizedSearchCV

Applied RandomizedSearchCV to Random Forest.

**Parameter search space:**
- `n_estimators`: [5, 50, 100]
- `max_depth`: [5, 10, 15, 20]

The search used:
- 7-fold cross validation
- 3 parameter combinations (`n_iter=3`)

**Best parameters from the recorded notebook output:**

```python
{
    "n_estimators": 100,
    "max_depth": 15
}
```

## Key Learnings

Through this project, I practiced:

- Data preprocessing and exploratory data analysis.
- Handling categorical variables using one-hot encoding.
- Feature scaling using StandardScaler.
- K-Fold Cross Validation.
- Stratified K-Fold Cross Validation.
- Comparing classification models using cross-validation accuracy.
- Hyperparameter optimization using GridSearchCV.
- Hyperparameter optimization using RandomizedSearchCV.
- Understanding how model parameters influence performance.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Repository Structure

```text
Hotel-Reservation-Cancellation-Prediction/
│
├── Hotel_Reservation_Cancellation_Prediction.ipynb
├── hotel_reservations.csv
├── README.md
└── requirements.txt
```

## How to Run This Project

1. Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

2. Navigate to the project directory:

```bash
cd Hotel-Reservation-Cancellation-Prediction
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Make sure `hotel_reservations.csv` is in the same directory as the notebook.

5. Open Jupyter Notebook:

```bash
jupyter notebook
```

6. Open `Hotel_Reservation_Cancellation_Prediction.ipynb` and run the cells.

## Future Improvements

- Use Scikit-learn Pipelines to prevent data leakage during cross validation.
- Apply preprocessing separately within each training fold.
- Explore additional evaluation metrics such as precision, recall, F1-score, and ROC-AUC.
- Expand the hyperparameter search space.
- Compare tuned models on a separate, untouched test set.
- Evaluate model performance with a confusion matrix.

## Important Evaluation Note

In the current notebook, `StandardScaler` is fitted on the full dataset before cross validation. This allows information from validation folds to influence preprocessing and can lead to data leakage.

A useful next improvement is to place scaling and model training inside a Scikit-learn Pipeline so that the scaler is fitted separately within each training fold.

The reported cross-validation scores are the results recorded in the notebook and should be interpreted with this limitation in mind.

## Acknowledgment

Dataset credit: Ahsan Raza.

[Hotel Reservations Classification Dataset – Kaggle](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)

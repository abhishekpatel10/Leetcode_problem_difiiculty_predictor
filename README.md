# LeetCode Problem Difficulty Prediction

This project aims to predict the difficulty level (Easy, Medium, or Hard) of LeetCode problems using various machine learning models. The prediction is based on features extracted from the problem's metadata, including its acceptance rate, submission statistics, topic tags, and the problem statement itself.

## 📋 Table of Contents
* [Project Overview](#-project-overview)
* [Dataset](#-dataset)
* [Project Workflow](#-project-workflow)
* [Technologies & Libraries Used](#-technologies--libraries-used)
* [Installation & Usage](#-installation--usage)
* [Models Implemented](#-models-implemented)
* [Evaluation & Results](#-evaluation--results)
* [Feature Importance](#-feature-importance)
* [Contributing](#-contributing)
* [License](#-license)

## 📌 Project Overview

The primary goal is to build a multi-class classification system that accurately categorizes LeetCode problems. The project demonstrates a complete machine learning pipeline, including:

1.  **Data Cleaning & Preprocessing**: Handling missing values and preparing categorical and numerical data for modeling.
2.  **Exploratory Data Analysis (EDA)**: Visualizing feature distributions, relationships, and correlations to gain insights.
3.  **Feature Engineering**:
    *   Applying **TF-IDF Vectorization** to convert the text of problem statements into numerical features.
    *   Using **Standard Scaling** on numerical columns to normalize their range.
    *   Combining numerical, text, and categorical (topic tags) features into a single feature matrix.
4.  **Data Balancing**: Addressing class imbalance in the `difficulty` column by applying undersampling to the majority class.
5.  **Model Training & Tuning**: Implementing several classification algorithms and optimizing their performance using `GridSearchCV`.
6.  **Comprehensive Evaluation**: Assessing model performance using metrics like Accuracy, Precision, Recall, F1-Score, Confusion Matrices, ROC Curves, and Precision-Recall Curves.

## 💾 Dataset

The project uses a custom dataset named `leetcode_questions_processed1.csv`, which contains various attributes for each LeetCode problem:

*   **`question_id`**: Unique identifier for the problem.
*   **`title`**, **`slug`**: Descriptive names for the problem.
*   **`difficulty`**: The target variable (Easy, Medium, Hard).
*   **`category`**: The problem category (e.g., Algorithms, Database).
*   **`ac_rate`**: The acceptance rate of the problem.
*   **`total_accepted`**, **`total_submissions`**: Submission statistics.
*   **`topic_tags`**: Comma-separated list of relevant topic tags (e.g., Array, Hash Table).
*   **`problem_statement`**: The full text of the problem description.

##  workflow Project Workflow

1.  **Load Data**: The dataset is loaded using Pandas.
2.  **Exploratory Data Analysis (EDA)**: The data is explored to understand feature distributions, such as the number of problems per difficulty level and the most frequent topic tags.
3.  **Data Preprocessing & Feature Engineering**:
    *   The target variable `difficulty` is mapped from text labels to numerical values (Easy: 0, Medium: 1, Hard: 2).
    *   The dataset is balanced using random undersampling of the 'Medium' class.
    *   The data is split into training and testing sets.
    *   `StandardScaler` is applied to numerical features.
    *   `TfidfVectorizer` is used to process the `problem_statement`.
    *   All features are combined into a final sparse matrix for training.
4.  **Model Building**: Four different models are trained on the processed data.
5.  **Hyperparameter Tuning**: `GridSearchCV` with 5-fold repeated stratified cross-validation is used to find the best hyperparameters for each model.
6.  **Evaluation**: The performance of the best models is visualized using Confusion Matrices, ROC curves, and PR curves.
7.  **Analysis**: Feature importance is extracted from the top models to identify the most influential predictors.

## 💻 Technologies & Libraries Used

*   **Python 3.x**
*   **Pandas** & **NumPy**: For data manipulation and numerical operations.
*   **Matplotlib** & **Seaborn**: For data visualization.
*   **Scikit-learn**: For data preprocessing, model implementation, and evaluation.
*   **XGBoost**: For the XGBoost classification model.
*   **SciPy**: For handling sparse matrices.

## ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/abhishekpatel10/Leetcode_problem_difiiculty_predictor.git
    cd Leetcode_problem_difiiculty_predictor
    ```

2.  **Install the required libraries:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *If you don't have a `requirements.txt` file, you can create one or provide the manual installation command:*
    ```bash
    pip install pandas matplotlib seaborn numpy xgboost scikit-learn scipy
    ```

3.  **Run the Notebook**:
    Launch Jupyter Notebook or JupyterLab and open the `.ipynb` file to view and execute the code cells.
    ```bash
    jupyter notebook Milestone_3_ap3268_1.ipynb
    ```

## 🤖 Models Implemented

The following classification models were trained and evaluated:
*   **Logistic Regression**
*   **Random Forest Classifier**
*   **K-Nearest Neighbors (KNN)**
*   **XGBoost Classifier**

## 📊 Evaluation & Results

After hyperparameter tuning, the models were evaluated on the test set. **XGBoost** emerged as the best-performing model with the highest accuracy and a strong balance of precision and recall across all classes.

| Model                 | Test Accuracy | F1-Score (Macro Avg) | F1-Score (Weighted Avg) |
| :-------------------- | :-----------: | :------------------: | :---------------------: |
| **XGBoost**           |   **72.68%**  |       **0.73**       |        **0.73**         |
| Random Forest         |    67.93%     |         0.68         |          0.68           |
| Logistic Regression   |    66.98%     |         0.66         |          0.67           |
| K-Nearest Neighbors   |    60.57%     |         0.59         |          0.59           |

### Visualizations

The performance of the top models was further analyzed with the following plots:

*   **Confusion Matrix**: To visualize prediction accuracy for each difficulty class.
*   **ROC and Precision-Recall Curves**: To evaluate the trade-off between true positive rate and false positive rate, and precision vs. recall.



## 🔑 Feature Importance

Feature importance analysis from both Random Forest and XGBoost models revealed that the most significant predictors of problem difficulty are:
1.  **`ac_rate`**: The problem's acceptance rate was consistently the most powerful feature.
2.  **TF-IDF Features**: Words from the problem statement like `return`, `array`, `binary`, and `matrix` were highly influential.
3.  **Topic Tags**: Tags such as `Dynamic Programming`, `Array`, and `Math` also contributed significantly to the model's predictions.



## 🙌 Contributing

Contributions are welcome! If you have any ideas, suggestions, or find any bugs, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

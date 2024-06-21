1- Overview:
The Loan Approval Predictor project aims to automate the loan eligibility process for Dream Housing Finance by leveraging data science techniques and machine learning models. The project consists of exploratory data analysis (EDA), feature engineering, model training, and deployment of an interactive Streamlit application.

2- Exploratory Data Analysis (EDA):
- Conducted a comprehensive analysis of the provided dataset (train_ctrUa4K.csv).
- Addressed issues such as missing values, duplicates and outliers.
- Explored relationships between demographic and socioeconomic factors and loan approval rates.

3- Feature Engineering:
- Created new features to enhance the predictive power of the model.
- Engineered features such as total income, monthly loan payments and income after loan deduction.
- Utilized logarithmic transformations to normalize skewed distributions.

4- Model Training and Evaluation:
- Explored various classification models, including Logistic Regression, Gaussian Naive Bayes, SVM, Decision Tree, Random Forest and XGBoost.
- Implemented a pipeline for data preprocessing, including imputation, encoding, scaling and addressing class imbalance.
- Used cross-validation to evaluate model performance and fine-tuned the XGBoost model using GridSearchCV.

5- Deployment with Streamlit:
- Developed an interactive Streamlit application (streamlit_app.py) for users to predict loan approval.
- Integrated the final trained SVM model into the Streamlit app.
- Enabled users to input details such as gender, marital status, education and income to receive real-time loan approval predictions.

6- Usage
You can interact with the deployed application on the cloud by visiting the following link:
(https://loan-prediction-dream-housing-finance-project-j6t4w42ayehercbk.streamlit.app/)

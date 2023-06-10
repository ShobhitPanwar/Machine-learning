Machine-learning repository contains two branches:
1. main
2. model-optimised

In this repository, I will be buidling several ML model taking different datasets and problems.

File description:
1. breast_cancer_model_random_forest.py: I have used "Breast Cancer Wisconsin (Diagnostic)" dataset for analyzing breast cancer problem and built an ML model 
to detect breast cancer in females. In this python file, I have done EDA, data cleaning, normalization, built model using Random forest classifier, and checked 
performance of the model using accuracy as performance metric.

2. breast_cancer_dataset.csv: This is the Breast Cancer Wisconsin (Diagnostic) dataset being used in the above python file.

3. Alphabet Recognition with CNN (2) 2.ipynb: I have used EMNIST english handwritten dataset, and have built an CNN model using keras to recognize handwritten alphabets. Accuracy of this model is nearly 93%. I have also built an UI where user can make their own english alphabet and model will recognize it.

4. Email_program.ipynb: I have built a spam email detector application. In this application, user needs to first signup/login, user's credentials are encrypted before getting saved to database. Once logged in, user needs to enter the email, and model will predict weather it's spam or not. To interact with database, sqllite is been used in this application and for model building Multinomial Naive Bayes.

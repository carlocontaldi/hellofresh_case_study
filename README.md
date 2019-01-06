# HelloFresh Case Study
Data science case study on subscription churn rate, based on real-world HelloFresh data. The report includes an in-depth description of my work and thought process. This project has been carried out in 4 days (time allotted for the challenge).

# Prerequisites
I used Conda 4.5.11 with Python 3.6.5 on a machine implementing Windows 10 64-bit to set up my PyData stack. I worked within a new environment defined and activated via terminal as in the following:
`> conda create --name hellofresh python=3.6.5 numpy pandas matplotlib seaborn scikit-learn py-xgboost`
`> conda activate hellofresh`

The data used in this case study can be found @<https://drive.google.com/drive/folders/140jRpRUOGCWDLtg1q1GtEz53JINi4aUE?usp=sharing>. If you would like to reproduce my results download the data files, store them in a "data" folder and run the main script.

The `main.py` script executes the whole workflow and saves the output in a logfile.
`> python main.py`

# Highlights
- EDA: framed the problem and leveraged ambiguous features by thinking out of the box.
- Feature-engineered a new dataset more suitable to the task at hand and applied relevant models.
- My pipeline involved one-hot encoding, normalization, multicollinearity assessment, grid search, logistic regression and XGBoost with early stopping.
- My best model achieved 81.5% accuracy and performed statistically significantly better than the naïve baseline on the test set (61%).

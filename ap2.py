import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main():
    st.title("Score Predictor")

    st.header("Upload Training Data")
    train_file = st.file_uploader("Upload a CSV file with training data:", type=["csv"])

    if train_file is not None:
        train_df = pd.read_csv(train_file, encoding="utf-8")
        
        # Prepare the data
        final_scores = train_df['FINAL SCORE']
        train_df = pd.get_dummies(train_df, columns=['LEARNING STATUS'])
        X = train_df[['THINKING CAPACITY', 'LOGICAL THINKING', 'ANALYTICAL POWER', 'NUMERACY', 'STRESS RESISTANCE', 'ACHIEVEMENT MOTIVATION', 'CONTINUOUS LEARNING', 'TASK COMMITMENT', 'INTEGRITY', 'EMOTIONAL STABILITY', 'SELF-ADJUSTMENT', 'SOCIAL RELATIONS', 'EFFECTIVE COMMUNICATION', 'CO-OPERATION', 'PRETEST GRADE']]
        y = train_df[['LEARNING STATUS_PASS']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train the logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train.values.ravel())

        st.header("Upload New Student Data")
        new_student_file = st.file_uploader("Upload a CSV file with new student data:", type=["csv"])

        if new_student_file is not None:
            new_student_df = pd.read_csv(new_student_file, encoding="utf-8")
            new_student_df = pd.get_dummies(new_student_df, columns=['LEARNING STATUS'], dummy_na=True)

            # Predict the learning status and final score for the new student
            predictions = model.predict(new_student_df)
            probabilities = model.predict_proba(new_student_df)
            final_scores = (probabilities[:, 1] * 90) // 1
            learning_status = ['PASS' if score >= 61 else 'FAIL' for score in final_scores]

            # Add predictions to the new_student_df
            new_student_df['PREDICTED LEARNING STATUS'] = learning_status
            new_student_df['PREDICTED FINAL SCORE'] = final_scores

            st.write("Predicted Results:")
            st.write(new_student_df)

            # Visualize the comparison between actual scores and predicted scores
            fig, ax = plt.subplots()
            ax.scatter(final_scores, final_scores, c='black', label='Actual')
            ax.scatter(final_scores, new_student_df['PREDICTED FINAL SCORE'], c='red', label='Predicted')
            ax.legend()
            ax.set_xlabel("Actual Final Scores")
            ax.set_ylabel("Predicted Final Scores")
            st.pyplot(fig)

if __name__ == "__main__":
    main()

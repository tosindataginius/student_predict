# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 14:56:43 2025

@author: tosindataginius
"""

import streamlit as st
import numpy as np 
import pickle

 #loading the saved model
model = pickle.load(open('C:/Users/USER/Downloads/MachineLearning/Student_Dropout_Prediction.pkl','rb'))


# Creating a function for prediction
def dropout_prediction(new_data):
    # Check if the model loaded successfully
    if model is None:
        return 'Prediction failed due to model loading error.'

    # Input validation and conversion to numeric
    try:
        numeric_data = [float(x) for x in new_data]
    except ValueError:
        return 'Error: All input fields must contain numbers.'
    
    # changing the input to numpy array
    input_data_as_numpy_array = np.asarray(numeric_data)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)  
    
    
    
    # Perform prediction
    result = model.predict(input_data_reshaped)
    print(result) # For local logging/debugging
    
    # Assuming 0 is good standing and 1 is dropout
    if (result[0] == 0):
        return "LOW RISK (Predicted to Remain)"
     
    else:
        return "HIGH RISK (Predicted to Dropout)"                             
       
   
    


def main():
    
    st.title("🎓 Student Dropout Prediction Dashboard")
    st.markdown("Enter the student's key academic data to predict their risk of dropping out.")

# --- 3. INPUT WIDGETS (Sidebar for cleaner layout) ---

    st.sidebar.header("Input Student Features")

    
    # getting the input from the user
   
    
    # 1. JAMB Score (This is a numerical score, e.g., max 400)
    JAMPScore = st.sidebar.slider(
    "1. JAMB Score",
    min_value=100,
    max_value=400,
    value=200,
    step=1
    )

    # 2. Academic Year (Assuming a small integer representing the current year of study, e.g., 1 to 4)
    AcademicYear = st.sidebar.selectbox(
    "2. Academic Year",
    options=[1, 2, 3, 4, 5, 6],
    index=0 # Default to Year 1
    )

    # 3. Cumulative GPA (Assuming GPA is on a 5.0 scale)
    CumulativeGPA = st.sidebar.number_input(
    "3. Cumulative GPA (e.g., 0.0 to 5.0)",
    min_value=0.0,
    max_value=5.0,
    value=3.0,
    step=0.1,
    format="%.2f"
    )

    # 4. Attendance Rate (Percentage)
    AttendanceRate = st.sidebar.slider(
    "4. Attendance Rate (%)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    )

    
    
    
    # code for prediction
    output = ''
    
    # creating a button for prediction
    if st.button("Predict Dropout Risk", type="primary"):
        # Only attempt prediction if all fields are filled
        if all([JAMPScore, AcademicYear, CumulativeGPA, AttendanceRate]):
            output = dropout_prediction([JAMPScore, AcademicYear, CumulativeGPA, AttendanceRate])
        else:
            output = 'Please enter values for all fields.'

    st.success(output)
    

if __name__ == '__main__':
    main() 
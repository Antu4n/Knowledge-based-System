import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime

# Load the trained model
model = joblib.load('model.pkl')

road_mapping ={'Bomet   Sotik highway': 0, 'Eldoret Webuye Highway': 1, 'Homabay  Kendubay Road': 2, 'Isinya Kiserian Road': 3, 'Kakamega Kisumu Road': 4, 'Kenol Muranga road': 5, 'Kenol Sagana Road': 6,
            'Kerugoya Karatina Road': 7, 'Kilifi Malindi Road': 8, 'Kisii Keroka Road': 9, 'Kisumu   Busia Road': 10, 'Kisumu Highway': 11, 'Kisumu Kakamega road': 12, 
            'Lodwar Kakuma Road': 13, 'Maai Mahiu Naivasha Highway': 14, 'Marsabit Isiolo Highway': 15, 'Meru Embu Road': 16, 'Migori Isibania Road': 17, 'Migori Kisii Isebania Highway': 18,
            'Mombasa   Nairobi highway': 19, 'Nairobi   Nakuru highway': 20, 'Nairobi Express way': 21, 'Nairobi Kakamega road': 22, 'Naivasha Nairobi Highway': 23, 'Nakuru   Nairobi highway': 24, 
            'Nakuru Eldoret Highway': 25, 'Nakuru Eldoret highway': 26, 'Nakuru Kericho Highway': 27, 'Namanga Road': 28, 'Narok Mai Mahiu road': 29, 'Narok road': 30, 'Naromoru Nanyuki Road': 31, 
            'Rukenya Kimunye Road': 32, 'Sagana Kagio Road': 33, 'Sagana Kenol Road': 34, 'Thika Kitui Highway': 35, 'Thika Road': 36}

# Function to preprocess user input data 
def preprocess_data(time_of_travel):
    try:
        # Convert time to the correct format
        time_obj = datetime.strptime(time_of_travel.strip(), "%H:%M")
        hour = time_obj.hour #Extract hour from time string    
        # apply the scaling
        scaler = StandardScaler()
        scaled_hour = scaler.fit_transform([[hour]])  # Reshape input as 2D array
        return scaled_hour
    except ValueError:
        st.error("Invalid time format. Please enter the tim in HH:MM format (eg. 13:30)")
        return None

# Create the app interface
def main():
    st.title("Road Safety Prediction")
    st.write("Enter the details below to check if the road is safe:")
    
    # User input: Time of travel (could be any other features you want to use)
    time_of_travel = st.text_input("Enter Time of Travel (e.g., 12:30, 18:45):", "12:00")
    road = st.selectbox("Select the Road/Highway:", road_mapping.keys())
    
    # Preprocess input data
    hour = preprocess_data(time_of_travel)
    
    # Predict if the road is safe or not based on the model
    if st.button('Predict Safety'):
        if hour is None:
            st.error("Invalid Input")
        else:
            input_data = pd.DataFrame({
                "Total people confirmed dead":[0],
                "Hour":[hour],
                "Road/Highway encoded":[road_mapping[road]]
            })
            prediction = model.predict(input_data)[0]
            
            if prediction == 1:
                st.success(f"**{road}** is safe at {time_of_travel}.")
            else:
                st.error(f"**{road}** is **unsafe** at {time_of_travel}.")
    
if __name__ == '__main__':
    main()

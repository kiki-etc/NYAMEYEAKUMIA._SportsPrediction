import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
model_path = "GradientBoostingRegressor.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

scaler_path = "scaler.pkl"
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Main program for Streamlit to use
def main():
    st.title("Predicting Player Ratings")
    html_temp = """
    <div style="background:#B6001B;padding:10px">
    <h2 style="color:#FFB6C1;text-align:center;">Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input fields...
    age = st.number_input('Age:')
    cm = st.number_input('CM:', 1, 100, 1)
    gk = st.number_input('GK:', 1, 100, 1)
    movement_reactions = st.number_input('Movement Reactions:')
    value_eur = st.number_input('Player Value (Euros):')
    wage_eur = st.number_input('Player Wage (Euros):')
    mentality_composure = st.number_input('Mentality Composure:', 1, 100, 1)
    power_shot_power = st.number_input('Shot Power:', 1, 100, 1)
    mentality_vision = st.number_input('Mentality Vision', 1, 100, 1)
    potential = st.number_input('Potential:', 1, 100, 1)

    if st.button('Predict'):
        # Gather input data...
        data = {
            'potential': [potential],
            'value_eur': [value_eur],
            'age': [age],
            'cm': [cm],
            'movement_reactions': [movement_reactions],
            'gk': [gk],
            'wage_eur': [wage_eur],
            'mentality_vision': [mentality_vision],
            'mentality_composure': [mentality_composure],
            'power_shot_power': [power_shot_power]
        }
        # Scale input data
        scaled_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))

        # Making into a DataFrame
        df = pd.DataFrame(scaled_data, columns=data.keys())
        
        # Ensure the DataFrame has the same columns as the model expects
        expected_features = model.feature_names_in_
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0  # or some default value

        df = df[expected_features]  # Reorder columns to match model's expectation
        
        prediction = model.predict(df)
        st.write("The predicted overall for your player is ", prediction[0])

if __name__ == '__main__':
    main()

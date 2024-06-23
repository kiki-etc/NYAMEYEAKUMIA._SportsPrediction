import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Load model and scaler
with open('GradientBoostingRegressor.pkl', 'wb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'wb') as f:
    scale = pickle.load(f)

# Define the expected feature names explicitly if not present in the model
try:
    expected_features = model.feature_names_in_
except AttributeError:
    expected_features = ['potential', 'value_eur', 'age', 'cm', 'movement_reactions',
                         'gk', 'wage_eur', 'mentality_vision', 'mentality_composure',
                         'power_shot_power']

def main():
    st.title("Predicting Player Ratings")
    html_temp = """
    <div style="background:#B6001B;padding:10px">
    <h2 style="color:#FFB6C1;text-align:center;">Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

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
        data = {
            'potential': potential,
            'value_eur': value_eur,
            'age': age,
            'cm': cm,
            'movement_reactions': movement_reactions,
            'gk': gk,
            'wage_eur': wage_eur,
            'mentality_vision': mentality_vision,
            'mentality_composure': mentality_composure,
            'power_shot_power': power_shot_power
        }

        # Creating DataFrame in correct order
        df = pd.DataFrame([data], columns=expected_features)
        
        # Scaling input data
        scaled_data = scale.transform(df)
        
        # Creating a scaled DataFrame with expected features
        scaled_df = pd.DataFrame(scaled_data, columns=expected_features)

        prediction = model.predict(scaled_df)
        st.write("The predicted overall for your player is ", prediction[0])

if __name__ == '__main__':
    main()

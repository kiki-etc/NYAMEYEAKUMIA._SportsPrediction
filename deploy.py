import streamlit as st
import pandas as pd
import numpy as np
import pickle

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="Yaaba/Training-Model", filename="GradientBoostingRegressor.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

scale_path = hf_hub_download(repo_id="Yaaba/Training-Model", filename="scaler.pkl")
with open(model_path, 'rb') as f:
    scale = pickle.load(f)

# Main program for Streamlit to use
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
        df = pd.DataFrame([data], columns=data.keys())
        
        # Scaling input data
        scaled_data = scale.transform(df)
        
        # Ensure the DataFrame has the same columns as the model expects
        expected_features = model.feature_names_in_
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
        scaled_df = scaled_df[expected_features]

        prediction = model.predict(scaled_df)
        st.write("The predicted overall for your player is ", prediction[0])

if __name__ == '__main__':
    main()
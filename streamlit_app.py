import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load the model and label encoders
sp_model = joblib.load('sp_election_model.pkl')
le_state = joblib.load('le_state.pkl')
le_candidate = joblib.load('le_candidate.pkl')
le_party_detailed = joblib.load('le_party_detailed.pkl')

# Load datasets
sp_president_df = pd.read_csv('1976-2020-president.csv', encoding='latin1')
sp_senate_df = pd.read_csv('1976-2020-senate.csv', encoding='latin1')

# Function to preprocess input data for prediction
def preprocess_input(year, state, candidate, party_detailed, totalvotes):
    if state == '' or candidate == '' or party_detailed == '':
        raise ValueError("All fields must be filled")
    
    input_data = pd.DataFrame({
        'year': [year],
        'state': [state],
        'candidate': [candidate],
        'party_detailed': [party_detailed],
        'totalvotes': [totalvotes]
    })
    
    input_data['state'] = le_state.transform(input_data['state'])
    input_data['candidate'] = le_candidate.transform(input_data['candidate'])
    input_data['party_detailed'] = le_party_detailed.transform(input_data['party_detailed'])
    return input_data

# Streamlit App
st.title("Election Outcome Prediction")

st.sidebar.header("Input Parameters")
year = st.sidebar.number_input("Year", min_value=1976, max_value=2020, step=4)
state = st.sidebar.selectbox("State", [''] + list(sp_president_df['state'].unique()))
candidate = st.sidebar.text_input("Candidate")
party_detailed = st.sidebar.selectbox("Party Detailed", [''] + list(sp_president_df['party_detailed'].unique()))
totalvotes = st.sidebar.number_input("Total Votes", min_value=0, step=1000)

# Ensure all fields are filled before processing
if st.sidebar.button("Predict"):
    try:
        input_data = preprocess_input(year, state, candidate, party_detailed, totalvotes)
        prediction = sp_model.predict(input_data)
        prediction_proba = sp_model.predict_proba(input_data)

        st.write(f"Predicted Party: {prediction[0]}")
        st.write("Prediction Probability:")
        st.write(prediction_proba)
    except ValueError as e:
        st.error(e)

# Display USA map and state-wise data
st.subheader("Election Results Visualization")
state_data = sp_president_df.groupby('state_po')['candidatevotes'].sum().reset_index()
fig = px.choropleth(state_data, locations='state_po', locationmode="USA-states", color='candidatevotes', scope="usa")
st.plotly_chart(fig)

# Display other charts
st.subheader("Additional Charts")
fig_votes = px.bar(sp_president_df, x='year', y='totalvotes', color='party_simplified', title="Votes by Year and Party")
st.plotly_chart(fig_votes)

fig_candidates = px.bar(sp_president_df, x='candidate', y='candidatevotes', color='party_simplified', title="Votes by Candidate")
st.plotly_chart(fig_candidates)

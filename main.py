
import streamlit as st
import pandas as pd
import langchain
from langchain.llms import GooglePalm
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
import google.generativeai as palm

import api

palm.configure(api_key='AIzaSyD1hj2THn19J8gkaBFaTeOriokbD6e_oYU')
st.sidebar.title("Upload CSV File")
st.sidebar.markdown('catatan : hanya berisi 1 atribut teks')
file = st.sidebar.file_uploader("Choose a file")
if file is not None:
    csv_file = file
else:
    csv_file = None

df = pd.read_csv(file)

defaults = {
  'model': 'models/text-bison-001',
  'temperature': 0.5,
  'candidate_count': 1,
  'top_k': 40,
  'top_p': 0.95,
  'max_output_tokens': 1024,
  'stop_sequences': [],
  'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":"BLOCK_LOW_AND_ABOVE"},{"category":"HARM_CATEGORY_TOXICITY","threshold":"BLOCK_LOW_AND_ABOVE"},{"category":"HARM_CATEGORY_VIOLENCE","threshold":"BLOCK_MEDIUM_AND_ABOVE"},{"category":"HARM_CATEGORY_SEXUAL","threshold":"BLOCK_MEDIUM_AND_ABOVE"},{"category":"HARM_CATEGORY_MEDICAL","threshold":"BLOCK_MEDIUM_AND_ABOVE"},{"category":"HARM_CATEGORY_DANGEROUS","threshold":"BLOCK_MEDIUM_AND_ABOVE"}],
}
satu = df.columns[0]
df = df.rename(columns={satu: 'text'})
df.dropna(inplace=True)
df=df.head(50)
defaults = {
    'model': 'models/text-bison-001',
    'temperature': 0.5,
    'candidate_count': 1,
    'top_k': 40,
    'top_p': 0.95,
    'max_output_tokens': 1024,
    'stop_sequences': [],
    'safety_settings': [
        {"category": "HARM_CATEGORY_DEROGATORY", "threshold": "BLOCK_LOW_AND_ABOVE"},
        {"category": "HARM_CATEGORY_TOXICITY", "threshold": "BLOCK_LOW_AND_ABOVE"},
        {"category": "HARM_CATEGORY_VIOLENCE", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUAL", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_MEDICAL", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ],
}

# Iterate through DataFrame and perform sentiment analysis
for index, row in df.iterrows():
    prompt = f"Tell me whether the following sentence's sentiment is positive or negative or neutral: {row['text']}"
    response = palm.generate_text(**defaults, prompt=prompt)
    df.at[index, 'sentiment'] = response.result
st.title("Analisis Sentiment Berhasil!")
st.dataframe(df, use_container_width=True)
st.sidebar.markdown("### Download CSV")
    st.sidebar.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name='downloaded_data.csv',
        key='download_button'

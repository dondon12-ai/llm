
import streamlit as st
import langchain
from langchain.llms import GooglePalm
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
import api

st.sidebar.title("Upload CSV File")

file = st.sidebar.file_uploader("Choose a file")
if file is not None:
    csv_file = file
else:
    csv_file = None

df = pd.read_csv(csv_file)
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

def generate_response(question, df):
    """Generates a response to a question using the CSV agent."""
    agent = create_csv_agent(
        GooglePalm(temperature=0.5, google_api_key=api.api_key),
        df,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    response = agent.run(question)
    return response

# Main area for input and display
st.title("Ask me anything about your data!")
question = st.text_input("Enter your question:")

if question:
    response = generate_response(question, df)
    st.write("Response:")
    st.info(response)
else:
    st.write("Please enter a question to get a response.")

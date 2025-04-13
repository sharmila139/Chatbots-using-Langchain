from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import ollama
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import openai


import os
from dotenv import load_dotenv  
load_dotenv()

##Langsmith tracking

os.environ['LANGCHAIN_API_KEY']= os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with OLLAMA"

import os
from dotenv import load_dotenv 
load_dotenv()

#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant, please respond to the queries to your best knowledge"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,llm,temperature,max_tokens):
    llm= ChatOpenAI(model=engine)
    output_parser = StrOutputParser()
    chain = prompts | llm | output_parser
    answer = chain.invoke({'question':question})
    return answer

##playing with streamlit

st.title('Enhanced Q&A chatbot with OLLAMA')

#sidebar for settings
api_key=st.sidebar.text_input("Enter your OpenAI API key:", type="password")

#drop down to select various LLM Models
llm=st.sidebar.selectbox("Select an OpenAI model",["mistral"])

#adjusting response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

##main interface for user input
st.write("go ahead and ask any question")
user_input= st.text_input("You:")

if user_input:
    response= generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("please provide the input")


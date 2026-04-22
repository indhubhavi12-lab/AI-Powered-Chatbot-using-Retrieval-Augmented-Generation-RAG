import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("MODEL_NAME")

client = Groq(api_key=api_key)
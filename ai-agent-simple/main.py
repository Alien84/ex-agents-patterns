# scripts/lesson_01/script1.py
import os
from pathlib import Path
from dotenv import load_dotenv
import langgraph
from langchain_openai import ChatOpenAI

# Load .env from project root
env_path = Path(__file__).parents[2] / '.env'
load_dotenv(env_path)

llm = ChatOpenAI(model="gpt-4o-mini") 
response = llm.invoke("Hello! Are you working?") 
print(response.content)
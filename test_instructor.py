import os
import logging
from pydantic import BaseModel
import instructor
from openai import AzureOpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)
load_dotenv()

class TestModel(BaseModel):
    name: str
    age: int

client = instructor.from_openai(AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
))

print("Calling instructor...")
try:
    resp = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
        messages=[{"role": "user", "content": "Extract: John is 25"}],
        response_model=TestModel,
        max_completion_tokens=1000
    )
    print("Success!", resp)
except Exception as e:
    print("Error:", e)

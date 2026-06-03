
from langchain_openai import ChatOpenAI
import httpx

file_path = r'C:\Users\GenAIBLRANCUSR50\Downloads\test\.venv\sample_app.log'

with open(file_path, "r", encoding="utf-8") as f:
    log_data = f.read()

prompt = f"""

Analyze this application log.

Provide:
1. Executive Summary
2. Errors
3. Warnings
4. Root Causes
5. Recommendations

Log:
{log_data}

"""


client = httpx.Client(verify=False)

llm = ChatOpenAI( 
base_url="https://genailab.tcs.in" ,
model = "azure_ai/genailab-maas-Phi-4-reasoning", 
api_key="sk-OHahpUWaFfOIcJZflRMF6w",
http_client = client 
) 
response = llm.invoke(prompt)   

print(response.content)
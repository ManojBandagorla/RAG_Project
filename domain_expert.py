
from langchain_openai import ChatOpenAI
import httpx

input_prompt = f'''
Role : You are a domain expert in python development
Context : Developement of sample python code 
output : fibonacci series consists of minimum 20 numbers
'''

client = httpx.Client(verify=False)

llm = ChatOpenAI( 
base_url="https://genailab.tcs.in" ,
model = "azure_ai/genailab-maas-Llama-4-Maverick-17B-128E-Instruct-FP8", 
api_key="sk-OHahpUWaFfOIcJZflRMF6w",
http_client = client 
)
response = llm.invoke(input_prompt)

print(response.content)

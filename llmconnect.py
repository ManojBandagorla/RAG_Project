
from langchain_openai import ChatOpenAI
import httpx

input_prompt = input('I am ready to help you. If you have any questions please ask me :\n ')

client = httpx.Client(verify=False)

llm = ChatOpenAI( 
base_url="https://genailab.tcs.in" ,
model = "azure_ai/genailab-maas-Llama-4-Maverick-17B-128E-Instruct-FP8", 
api_key="sk-OHahpUWaFfOIcJZflRMF6w",
http_client = client 
)
response = llm.invoke(input_prompt)

print(response.content)

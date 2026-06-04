import requests
import httpx

# Disable SSL verification (your current workaround)
original_get = requests.get

def patched_get(*args, **kwargs):
    kwargs["verify"] = False
    return original_get(*args, **kwargs)

requests.get = patched_get

# HTTPX client
def get_http_client():
    return httpx.Client(verify=False)
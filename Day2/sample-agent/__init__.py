# from . import agent

import requests
# Fetch latest USD conversion rates and print the conversion_rates object
response = requests.get(
	"https://v6.exchangerate-api.com/v6/0582a90af241c7eb38b0c9a6/latest/USD",
    timeout=10
)

response.raise_for_status()
data = response.json()
print(data["conversion_rates"])
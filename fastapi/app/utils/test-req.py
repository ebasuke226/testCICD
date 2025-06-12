import requests
url = "https://www.googleapis.com/discovery/v1/apis/drive/v3/rest"
response = requests.get(url)
print("Status code:", response.status_code)
doc = response.json()
print(doc)

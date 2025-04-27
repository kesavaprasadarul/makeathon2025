import requests
import json

url = "https://pixeltranslate.kesava.lol/pixel-to-latlon/batch"

payload = json.dumps({
  "imageName": ["DJI_20250424193302_0154_V.jpeg"],
  "u": [
    4031,
    4031
  ],
  "v": [
    3023,
    3020
  ],
  "datasource_name": "regensburg"
})
headers = {
  'accept': 'application/json',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
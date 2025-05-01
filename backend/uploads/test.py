import requests
import sys

# Define the file to upload
file_path="f64fcdf4-99c1-4e58-bfc9-cf2f7b4f0c27_20250501000642_noodles_46.jpg"

"""
# Make the POST request
response = requests.post(url, headers=headers, files=files)

# Print the response
print(response.status_code)
print(response.json())

import requests
import sys
"""
url =  "https://8000-m-s-l9jil7pzal0f-b.us-west4-1.prod.colab.dev/predict/"


# Define headers
headers = {
    'accept': 'application/json'
}

# Path to your image file
#file_path = 'noodles_46.jpg'

try:
    # Define the file to upload
    with open(file_path, 'rb') as f:
        files = {
            'file': (file_path, f, 'image/jpeg')
        }

        print(f"Sending request to: {url}")

        # Make the POST request
        response = requests.post(url, headers=headers, files=files, timeout=10)

        # Print detailed response information
        print(f"Status code: {response.status_code}")
        print(f"Response headers: {response.headers}")

        if response.status_code == 200:
            print("Response JSON:")
            print(response.json())
        else:
            print(f"Error response text: {response.text}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except requests.exceptions.ConnectionError:
    print(f"Error: Could not connect to the server. The URL might be incorrect or the server is down.")
except requests.exceptions.Timeout:
    print(f"Error: The request timed out. The server might be slow or unresponsive.")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
    print(f"Error type: {type(e).__name__}")

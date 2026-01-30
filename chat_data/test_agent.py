import requests
import os
import time

URL = "http://localhost:8000"
THREAD_ID = f"session_{int(time.time())}" 


csv_content = """date,sales,expenses,region
2023-01-01,100,80,North
2023-01-02,150,90,North
2023-01-03,200,110,South
2023-01-04,130,100,South
2023-01-05,400,150,East
"""
with open("test_data.csv", "w") as f:
    f.write(csv_content)

print("--- Step 1: Uploading CSV ---")
try:
    with open("test_data.csv", "rb") as f:
        r_upload = requests.post(f"{URL}/upload", files={"file": f})
    
    if r_upload.status_code != 200:
        print(f"Upload Failed: {r_upload.text}")
        exit()

    file_path = r_upload.json()["file_path"]
    print(f"File uploaded successfully to: {file_path}")

    print("\n--- Step 2: Asking a statistical question ---")
    payload = {
        "message": "Which region had the maximum sales? Please show your reasoning.",
        "thread_id": THREAD_ID,
        "file_path": file_path
    }
    r_chat1 = requests.post(f"{URL}/chat", data=payload)
    print(f"Agent Response: {r_chat1.json().get('response', 'No response field found')}")

    print("\n--- Step 3: Asking for a plot ---")
    payload["message"] = "Create a bar chart showing sales by region and save it."
    r_chat2 = requests.post(f"{URL}/chat", data=payload)
    data = r_chat2.json()
    
    print(f"Agent Response: {data.get('response')}")
    if data.get('plot_url'):
        print(f"SUCCESS: Plot available at: {URL}{data['plot_url']}")
    else:
        print("WARNING: No plot URL returned.")

except requests.exceptions.ConnectionError:
    print(f"ERROR: Could not connect to the server at {URL}. Is Docker running?")
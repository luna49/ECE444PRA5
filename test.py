import requests
import time
import csv
import matplotlib.pyplot as plt

test_cases = [
    {"text": "The world ended"},  # Fake news 1
    {"text": "AMD is not a real company"},   # Fake news 2
    {"text": "The economy is growing according to recent reports"},  # Real news 1
    {"text": "Canada is a country"}  # Real news 2
]

url_load_model = 'ECE444-env-1.eba-qxhskvz8.ca-central-1.elasticbeanstalk.com/load_model'
url_predict = 'ECE444-env-1.eba-qxhskvz8.ca-central-1.elasticbeanstalk.com/predict'

response = requests.post(url_load_model)
if response.status_code == 200:
    print("Model loaded successfully.")
else:
    print("Error loading model:", response.json())
    exit(1)

num_calls = 100

latency_results = {}

for i, test_case in enumerate(test_cases):
    latencies = []
    for _ in range(num_calls):
        start_time = time.time() # start time
        response = requests.post(url_predict, json=test_case)
        end_time = time.time()  # end time
        latency = end_time - start_time 
        latencies.append(latency)

        with open(f'latency_case_{i+1}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([latency])

    latency_results[f"case_{i+1}"] = latencies

plt.figure(figsize=(10, 6))
plt.boxplot([latency_results["case_1"], latency_results["case_2"], 
             latency_results["case_3"], latency_results["case_4"]])
plt.xticks([1, 2, 3, 4], ['Fake News 1', 'Fake News 2', 'Real News 1', 'Real News 2'])
plt.title('Latency Boxplot')
plt.ylabel('Latency (seconds)')
plt.show()

for i in range(4):
    avg_latency = sum(latency_results[f"case_{i+1}"]) / num_calls
    print(f"Average latency for case_{i+1}: {avg_latency} seconds")

# PLUMBING -CLIENT SERVICE CONNECTION APPLICATION SYSTEM -THRUMP FIX
# Core Feature 1: Location-Based Plumber Discovery
# Core Feature 2: Job Posting and Acceptance
# ===============================================================
# DESIGN LOGIC.
# 1.Client submits request.
# 2.System classifies the problem category.
# 3.System predicts severity(Using Logistic Regression).
# 4.System filters available plumbers within the same zone.
# 5.System calculates distance of available plumber to client using live coordinates.
# 6.System assigns nearest plumber.
# 7. For multiple requests; system assigns rule based classification and attends to the requests in order of severity.


#IMPORTING REQUIRED LIBRARIES
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from geopy.distance import geodesic

#LOADING  THE DATASETS.

plumber_df = pd.read_csv('C:/Users/ADMIN/Downloads/CAPSTONE PROJECT/Datasets/Cleaned_Plumber.csv')
client_df = pd.read_csv('C:/Users/ADMIN/Downloads/CAPSTONE PROJECT/Datasets/Cleaned_Client_Request.csv')
zones_df = pd.read_excel('C:/Users/ADMIN/Downloads/CAPSTONE PROJECT/Datasets/Zones Datasets..xlsx')

#Exploring the datasets we have.
print ('Datasets loaded Successfully!')
print (plumber_df.shape)
print (client_df.shape)
print (zones_df.shape)

#PREPARING DATA FOR MACHINE LEARNING-
#Severity Categorization.

def assign_severity(description):
    description = description.lower()

    high_keywords = [
        'burst', "flood", "overflow", "sewer", "backup",
        "collapsed", "explosion", "gas leak",
        "water main", "severely damaged",
        "affecting multiple", "hospital", "school",
        "traffic", "public area"
    ]

    medium_keywords = [
        "clog", "blocked", "leak", "leaking",
        "low pressure", "not working",
        "cracked", "damaged", "faulty",
        "no hot water", "pipe issue"
    ]

    low_keywords = [
        "install", "installation",
        "add", "replace", "new",
        "upgrade", "maintenance",
        "renovation", "improvement",
        "service request"
    ]

    if any(word in description for word in high_keywords):
        return "High"

    elif any(word in description for word in medium_keywords):
        return "Medium"

    elif any(word in description for word in low_keywords):
        return "Low"

    else:
        return "Low"

client_df["Severity"] = client_df["Description of Need"].apply(assign_severity)
print(client_df["Severity"].value_counts())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(client_df["Description of Need"])
print("Number of unique words:", len(vectorizer.get_feature_names_out()))
print(X.shape)

#Training our Model(Logistic Regression).
# Encoding severity labels
encoder = LabelEncoder()
y = encoder.fit_transform(client_df["Severity"])

# Split the TF-IDF features and labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print('Training successful.The Model has learned the patterns.')

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model accuracy:", round(accuracy * 100, 2), "%")

# #PLUMBER MATCHING AND ASSIGNMENT LOGIC
# # Creating Object-Oriented Programming Structure
# from geopy.distance import geodesic

class Plumber:

    def __init__(self, certificate_id, latitude, longitude, zone, available=True):
        self.certificate_id = certificate_id
        self.location = (latitude, longitude)
        self.zone = zone
        self.available = available

    def accept_job(self):
        self.available = False
        print(f"Plumber with Certificate ID {self.certificate_id} has accepted the job.")

    def __repr__(self):
        return f"Plumber(CertificateID={self.certificate_id}, Zone={self.zone}, Available={self.available})"

class Dispatcher:

    def __init__(self, plumbers):
        self.plumbers = plumbers

    def find_nearest_plumber(self, client_location, zone):

        # Step 1: Filter by zone and availability
        available_plumbers = []

        for plumber in self.plumbers:
            if plumber.available and plumber.zone == zone:
                available_plumbers.append(plumber)

        if not available_plumbers:
            print("No available plumbers in this zone.")
            return None

        # Step 2: Find nearest plumber
        nearest = None
        min_distance = float("inf")

        for plumber in available_plumbers:
            distance = geodesic(client_location, plumber.location).km

            if distance < min_distance:
                min_distance = distance
                nearest = plumber

        return nearest


# #Converting  DataFrame to Plumber Objects.

plumbers_list = []

for _, row in plumber_df.iterrows():
    plumber = Plumber(
        certificate_id=row["Plumber Certificate ID"],
        latitude=row["Base Latitude"],
        longitude=row["Base Longitude"],
        zone=row["LGA Zone"],
        available=True
    )
    plumbers_list.append(plumber)

# ----------- Create Dispatcher -----------

dispatcher = Dispatcher(plumbers_list)

# CALLING ---(SIMULATING NEW CLIENT REQUEST) #
# #CORE FEATURE 1: Location based plumber Assigning.
# CORE FEATURE 2: JOB POSTING AND ACCEPTANCE
# Client Request Looped.

print("Simulating new client request...\n")

while True:

    print("\n--- New Client Request ---")

    # Step 1: Problem description
    new_problem_description = input(
        "Describe the plumbing issue (type 'exit' to quit): "
    )

    if new_problem_description.lower() == "exit":
        print("System shutting down...")
        break

    # Step 2: Ask for location
    try:
        latitude = float(input("Enter client latitude: "))
        longitude = float(input("Enter client longitude: "))
    except ValueError:
        print("Invalid coordinates. Please enter numbers.")
        continue

    client_location = (latitude, longitude)

    # Step 3: Ask for zone
    client_zone = input("Enter client zone: ")

    print("\nClient Problem Description:", new_problem_description)

    # Step 4: Predict severity
    new_vector = vectorizer.transform([new_problem_description])
    predicted_severity_num = model.predict(new_vector)
    predicted_severity_label = encoder.inverse_transform(predicted_severity_num)

    print("Predicted Severity:", predicted_severity_label[0])
    print("Client Location:", client_location)
    print("Client Zone:", client_zone)
    print()

    # Step 5: Find nearest plumber
    nearest_plumber = dispatcher.find_nearest_plumber(
        client_location,
        client_zone
    )

    # Step 6: Assign job
    if nearest_plumber:
        print("Nearest plumber found:", nearest_plumber.certificate_id)
        nearest_plumber.accept_job()
    else:
        print("No plumber could be assigned.")

# SIMULATING MULTIPLE CLIENT REQUESTS

print("\n--- Simulating API Batch Request ---\n")

requests = []

# Step 1: Ask how many requests are coming in
num_requests = int(input("Enter number of simultaneous client requests: "))

# Step 2: Collect request data dynamically
for i in range(num_requests):
    print(f"\nEntering details for Client {i+1}")

    description = input("Enter problem description: ")

    try:
        latitude = float(input("Enter client latitude: "))
        longitude = float(input("Enter client longitude: "))
    except ValueError:
        print("Invalid coordinates. Skipping this client.")
        continue

    zone = input("Enter client zone: ")

    requests.append({
        "description": description,
        "location": (latitude, longitude),
        "zone": zone
    })
for request in requests:
    vector = vectorizer.transform([request["description"]])
    pred = model.predict(vector)
    severity = encoder.inverse_transform(pred)[0]
    request["severity"] = severity

def severity_rank(severity):
    if severity == "High":
        return 3
    elif severity == "Medium":
        return 2
    else:
        return 1

requests = sorted(
    requests,
    key=lambda x: severity_rank(x["severity"]),
    reverse=True
)

print("\n--- Dispatching Based on Severity ---\n")

for i, request in enumerate(requests, start=1):

    print(f"Processing Client {i}")
    print("Description:", request["description"])
    print("Severity:", request["severity"])

    nearest_plumber = dispatcher.find_nearest_plumber(
        request["location"],
        request["zone"]
    )

    if nearest_plumber:
        print("Nearest plumber found:", nearest_plumber.certificate_id)
        nearest_plumber.accept_job()
    else:
        print("No available plumber. Request pending.")

    print("-" * 50)

# END OF PROGRAM

print("\nSystem execution completed successfully.")

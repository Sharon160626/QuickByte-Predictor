import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load dataset
data = pd.read_csv("restaurant_rush.csv")

# Step 2: Convert text labels to numbers
data['rush'] = data['rush'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Step 3: Split features and target
X = data[['hour', 'day', 'weekend']]
y = data['rush']

# Step 4: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 5: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Test accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Step 7: User input
hour = int(input("Enter hour (0-23): "))
day = int(input("Enter day (0=Mon, 6=Sun): "))
weekend = int(input("Weekend? (0/1): "))

input_data = pd.DataFrame([[hour, day, weekend]], columns=['hour', 'day', 'weekend'])
prediction = model.predict(input_data)

rush_map = {0: "Low", 1: "Medium", 2: "High"}
print("Predicted Rush:", rush_map[prediction[0]])

rush_map = {0: "Low", 1: "Medium", 2: "High"}
print("Predicted Rush:", rush_map[prediction[0]])
# Convert result back to text
rush_map = {0: "Low", 1: "Medium", 2: "High"}
print("Predicted Rush:", rush_map[prediction[0]])
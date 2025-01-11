import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: Load the CSV
df = pd.read_csv('output_data.csv')

# Step 2: Extract the features (coordinates) and the label
coordinates = df.iloc[:, 2:-1].values  # Extracting x1, y1, z1 to x33, y33, z33
labels = df['label'].values  # Extracting the 'label' column

# Step 3: Label Encoding for categorical labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(coordinates, encoded_labels, test_size=0.2, random_state=42)

# Step 5: Create and train the model (using Random Forest as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 7: Optionally, save the model
import joblib
joblib.dump(model, 'trained_model.pkl')

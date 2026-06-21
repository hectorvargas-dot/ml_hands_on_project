import pandas as pd

# Load the files
X = pd.read_csv("data/processed/raw_features/X_train.csv")
y = pd.read_csv("data/processed/target/y_train.csv")

# Concatenate y as the last column(s) of X
df = pd.concat([X, y], axis=1)

# Save the result
df.to_csv("data/processed/train_test_datasset.csv", index=False)
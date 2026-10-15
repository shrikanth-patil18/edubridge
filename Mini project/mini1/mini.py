import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("trades_orderbook_STUDENT_dirty.csv") 

# 2. Display dataset information
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# 3. Clean data
df = df.drop_duplicates()
df = df.dropna()

df["side"] = df["side"].str.strip().str.lower()

# 4. Remove invalid prices and quantities
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df = df.dropna(subset=["price", "quantity"])
df = df[(df["price"] > 0) & (df["quantity"] > 0)]

# 5. Create features
df["spread"] = df["ask_price_1"] - df["bid_price_1"]
df["mid_price"] = (df["ask_price_1"] + df["bid_price_1"]) / 2

df["imbalance"] = (
    (df["bid_qty_1"] - df["ask_qty_1"]) /
    (df["bid_qty_1"] + df["ask_qty_1"])
)

# 6. Visualize data
df["price"].plot(title="Trade Price")
plt.show()

df["side"].value_counts().plot.bar(title="Buy vs Sell")
plt.show()

# 7. Scale features
features = df[["price", "quantity", "spread", "imbalance"]]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

print("\nScaled data:\n", scaled_data[:5])

# 8. Save cleaned dataset
df.to_csv("trades_orderbook_clean.csv", index=False)

print("\nData cleaning completed!")
print("Cleaned dataset saved successfully.")
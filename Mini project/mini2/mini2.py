import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 1. Sample dataset
data = {
    "credit_score": [450, 500, 550, 600, 620, 650, 680, 700, 720, 750],
    "loan_amount": [500000, 400000, 600000, 300000, 400000,
                    250000, 350000, 200000, 300000, 250000],
    "eligible": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# 2. Input and output
X = df[["credit_score", "loan_amount"]]
y = df["eligible"]

# 3. Train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# 4. Get user input
score = float(input("Enter CIBIL credit score: "))
amount = float(input("Enter loan amount required: "))

# 5. Predict eligibility
prediction = model.predict([[score, amount]])

if prediction[0] == 1:
    print("Person is eligible for the loan")
else:
    print("Person is not eligible for the loan")

# 6. Plot graph
colors = df["eligible"].map({0: "red", 1: "green"})

plt.scatter(df["credit_score"], df["loan_amount"], c=colors)

plt.xlabel("CIBIL Credit Score")
plt.ylabel("Loan Amount Required")
plt.title("Loan Eligibility Prediction")
plt.legend(handles=[
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor="green", label="Eligible", markersize=8),
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor="red", label="Not Eligible", markersize=8)
])

plt.show()
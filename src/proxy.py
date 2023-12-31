import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_csv("./data/raw/ratings.dat", delimiter="::", header=None)
df.columns = ["UserID", "MovieID", "Rating", "Timestamp"]

accuracy = pd.read_csv("./data/predict_time.csv")
accuracy.rename(columns={"userId": "UserID"}, inplace=True)
accuracy.rename(columns={"acc": "Accuracy"}, inplace=True)

# Timestamp is in seconds

# Group the data by customer id and calculate the average time difference between reviews
grouped_data = df.groupby("UserID")

time_diffs = []

for _, group in grouped_data:
    if len(group) > 1:
        # Sort the group by rating date
        group = group.sort_values("Timestamp")

        # Calculate the time differences between consecutive reviews
        time_diff = group["Timestamp"].diff().mean()

        # Calculate the duration between the first and last review
        duration = group["Timestamp"].iloc[-1] - group["Timestamp"].iloc[0]

        time_diffs.append(
            {
                "UserID": group["UserID"].iloc[0],
                "avg_time_diff": time_diff,
                "duration": duration,
            }
        )
    else:
        # If the customer has only one review, set the average time difference and duration to 0.1
        time_diffs.append(
            {"UserID": group["UserID"].iloc[0], "avg_time_diff": 0, "duration": 0.1}
        )

avg_time_diff_df = pd.DataFrame(time_diffs)

# instead of just calculating the mean of the review time, account for users who are loyal (subscribed for a long time)
# new function -> mean time between reviews + lambda/(whole duration)

# lambda denoted by x that can be tuned to account for 'amount of time' a user's account have been registered
x = avg_time_diff_df["duration"].mean()  # 8207777.342880795
# x is a hyperparameter

avg_time_diff_df["metric"] = (
    avg_time_diff_df["avg_time_diff"] + x / avg_time_diff_df["duration"]
)

# the smaller the value of metric = more engaged users and users who have account registered for longer time (retained users)

merged_df = pd.merge(avg_time_diff_df, accuracy, on="UserID", how="inner")

# scatter plot
plt.scatter(merged_df["Accuracy"], merged_df["metric"])
plt.xlabel("Accuracy")
plt.ylabel("metric")
plt.title("Scatter Plot of Metric vs Accuracy")
plt.show()

# running OLS
X = merged_df["Accuracy"]
y = merged_df["metric"]

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

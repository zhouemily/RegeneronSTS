"""
The transformation is given by:

X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min
where min, max = feature_range.

The transformation is calculated as:

X_scaled = scale * X + min - X.min(axis=0) * scale
where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))
"""
from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
print("data=")
print(data)
scaler = MinMaxScaler()
print(scaler.fit(data))
print(scaler.data_max_)
print(scaler.transform(data))
print(scaler.transform([[2, 2]]))
print("data2=")
data2 = [[1, 2], [1, 6], [1, 10], [1, 18]]
print(data2)
scaler = MinMaxScaler()
print(scaler.fit(data2))
print(scaler.data_max_)
print(scaler.transform(data2))
print(scaler.transform([[1, 17]]))

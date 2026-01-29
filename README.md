# MCSL AI ML
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])

    visited.add(start)

    print("BFS Traversal:", end=" ")

    while queue:
        vertex = queue.popleft()
        print(vertex, end=" ")

        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

# Graph representation using dictionary
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

# Calling BFS function
bfs(graph, 'A')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of KNN:", accuracy)

def dfs(graph, node, visited):
    if node not in visited:
        print(node, end=" ")
        visited.add(node)

        for neighbour in graph[node]:
            dfs(graph, neighbour, visited)

# Graph representation using dictionary
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

visited = set()

print("DFS Traversal:", end=" ")
dfs(graph, 'A', visited)

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([40, 50, 60, 70, 80])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict
y_pred = model.predict(X)

print("Slope (m):", model.coef_)
print("Intercept (c):", model.intercept_)

# Plot
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.show()

import heapq

def a_star(graph, start, goal, heuristic):
    open_list = []
    heapq.heappush(open_list, (0, start))
    
    g_cost = {start: 0}
    parent = {start: None}

    while open_list:
        current_cost, current_node = heapq.heappop(open_list)

        if current_node == goal:
            path = []
            while current_node:
                path.append(current_node)
                current_node = parent[current_node]
            return path[::-1]

        for neighbor, cost in graph[current_node]:
            new_g = g_cost[current_node] + cost
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                f_cost = new_g + heuristic[neighbor]
                heapq.heappush(open_list, (f_cost, neighbor))
                parent[neighbor] = current_node

    return None

# Graph
graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 3)],
    'C': [('D', 1)],
    'D': []
}

# Heuristic values
heuristic = {
    'A': 4,
    'B': 2,
    'C': 2,
    'D': 0
}

path = a_star(graph, 'A', 'D', heuristic)
print("Optimal Path:", path)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create model
model = LogisticRegression(max_iter=5000)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Logistic Regression:", accuracy)

import math

def alphabeta(depth, nodeIndex, maximizingPlayer, values, alpha, beta):
    if depth == 3:
        return values[nodeIndex]

    if maximizingPlayer:
        bes

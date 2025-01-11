# Define a class to represent each item with value, weight, and value/weight ratio
class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight
        self.ratio = value / weight  # value-to-weight ratio

# Function to solve the fractional knapsack problem
def fractional_knapsack(capacity, values, weights):
    # Create a list of items with their value, weight, and ratio
    items = []
    for i in range(len(values)):
        items.append(Item(values[i], weights[i]))

    # Sort items by their value-to-weight ratio in descending order
    items.sort(key=lambda x: x.ratio, reverse=True)

    total_value = 0.0  # To store the total value of knapsack
    for item in items:
        if capacity == 0:
            break
        # If the item can be fully taken, take it
        if item.weight <= capacity:
            total_value += item.value
            capacity -= item.weight
        else:
            # Otherwise, take the fraction of the item
            total_value += item.value * (capacity / item.weight)
            capacity = 0  # Knapsack is full now

    return total_value

# Given data
values = [20, 30, 40, 32, 55]  # Values of the items (P1, P2, P3, P4, P5)
weights = [5, 8, 10, 12, 15]   # Weights of the items (W1, W2, W3, W4, W5)
capacity = 20                  # Maximum capacity of the knapsack

# Calculate the maximum value that can be carried in the knapsack
max_value = fractional_knapsack(capacity, values, weights)

print(f"The maximum value that can be obtained in the knapsack is: {max_value}")

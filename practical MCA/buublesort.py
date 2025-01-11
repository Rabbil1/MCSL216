# Function to implement Bubble Sort
def bubble_sort(arr):
    n = len(arr)
    num_comparisons = 0  # To count the number of comparison operations
    num_exchanges = 0  # To count the number of exchange operations
    outer_loop_iterations = 0  # To count the number of iterations of the outer loop
    inner_loop_iterations = 0  # To count the number of iterations of the inner loop

    # Outer loop for the number of passes
    for i in range(n):
        outer_loop_iterations += 1
        # Inner loop to perform comparisons and swaps
        for j in range(0, n - i - 1):
            inner_loop_iterations += 1
            num_comparisons += 1  # One comparison per inner loop iteration
            
            if arr[j] > arr[j + 1]:
                # Swap the elements
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                num_exchanges += 1  # Increment exchange counter after a swap

    return arr, num_comparisons, num_exchanges, outer_loop_iterations, inner_loop_iterations

# Input list
arr = [55, 25, 15, 40, 60, 35, 17, 65, 75, 10]

# Call the bubble_sort function
sorted_arr, comparisons, exchanges, outer_loops, inner_loops = bubble_sort(arr)

# Output the results
print("Sorted Array:", sorted_arr)
print("Number of comparisons:", comparisons)
print("Number of exchanges:", exchanges)
print("Number of outer loop iterations:", outer_loops)
print("Number of inner loop iterations:", inner_loops)

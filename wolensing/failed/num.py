import numpy as np

def generate_random_array_with_sum(n, target_sum):
    # Generate random numbers
    array = np.random.rand(n)
    # Normalize to make the sum equal to target_sum
    array = array / np.sum(array) * target_sum
    return array

# Example usage
n = 5  # Length of the array
target_sum = 10
result = generate_random_array_with_sum(n, target_sum)
print(result)
print("Sum:", np.sum(result))
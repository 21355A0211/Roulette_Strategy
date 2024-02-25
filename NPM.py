import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to generate 100 roulette numbers
def generate_roulette_numbers(n=100):
    return np.random.randint(0, 37, n)

# Function to predict the next 5 roulette numbers (for demonstration purposes)
def predict_next_numbers(roulette_numbers, n=5):
    mean = np.mean(roulette_numbers)
    std_dev = np.std(roulette_numbers)
    return  np.clip(np.random.normal(mean, std_dev, n).astype(int), 0, 36)

# Function to predict the next dozen and next column
def predict_next_dozen_and_column():
    next_dozen = np.random.randint(1, 4)
    next_column = np.random.randint(1, 4)
    return next_dozen, next_column

# Generate roulette numbers
roulette_numbers = generate_roulette_numbers()

# Create a pandas DataFrame
df = pd.DataFrame({'Roulette Numbers': roulette_numbers})

# Statistical analysis
most_frequent_number = df['Roulette Numbers'].mode().values[0]
least_frequent_number = df['Roulette Numbers'].value_counts().idxmin()

# Prediction of the next 5 numbers
next_numbers = predict_next_numbers(roulette_numbers, n=5)

# Prediction of the next dozen and next column
next_dozen, next_column = predict_next_dozen_and_column()

# Display the first 100 roulette numbers in a list
print("\nRoulette Numbers (First 100):")
print(roulette_numbers[:100])

# Display the statistical analysis
print(f'\nMost Frequent Number: {most_frequent_number}')
print(f'Least Frequent Number: {least_frequent_number}')
print(f'\nPrediction of the Next 5 Numbers (for demonstration):\n{next_numbers}')
print(f'Prediction of the Next Dozen: {next_dozen}')
print(f'Prediction of the Next Column: {next_column}')

# Plotting the histogram
plt.figure(figsize=(12, 6))

# Plot the histogram
plt.hist(df['Roulette Numbers'], bins=range(38), alpha=0.7, edgecolor='black', linewidth=1.2)
plt.axvline(x=most_frequent_number, color='red', linestyle='dashed', label=f'Most Frequent Number: {most_frequent_number}')
plt.axvline(x=least_frequent_number, color='blue', linestyle='dashed', label=f'Least Frequent Number: {least_frequent_number}')

# Customize the plot
plt.title('Histogram of Roulette Numbers (100 samples)')
plt.xlabel('Roulette Number')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('roulette_numbers_histogram_100_samples.png')

# Show the plot
plt.show()


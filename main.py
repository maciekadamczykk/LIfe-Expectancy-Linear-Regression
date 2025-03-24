import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = pd.read_csv('LifeExpectancy.csv')

#Split dataset into train set and test set
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

print(f"Training set size: {train_dataset.shape}")
print(f"Testing set size: {test_dataset.shape}")


column_name = "Life Expectancy"
if column_name not in dataset.columns:
    print(f"Column '{column_name}' not found in dataset!")
else:
    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(dataset[column_name], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Life Expectancy (Years)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Life Expectancy")
    plt.grid(True)
    plt.show()

    #Print statistical information
    mean_value = dataset[column_name].mean()
    median_value = dataset[column_name].median()
    std_dev = dataset[column_name].std()
    min_value = dataset[column_name].min()
    max_value = dataset[column_name].max()

    print("\nStatistical Summary of Life Expectancy:")
    print(f"Mean: {mean_value:.2f}")
    print(f"Median: {median_value:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Minimum: {min_value:.2f}")
    print(f"Maximum: {max_value:.2f}")

if "Country" not in dataset.columns:
    print("Required columns ('Country') not found in dataset!")
else:

    # Find top 3 countries with highest life expectancy
    top_countries = dataset[['Country', 'Life Expectancy']].sort_values(by='Life Expectancy', ascending=False).head(3)

    # Print results
    print("Top 3 countries with highest life expectancy:")
    print(top_countries)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def load_data_and_analyze():
    """
    Loads, explores, analyzes, and visualizes the Iris dataset.
    """
    try:
        # Task 1: Load and Explore the Dataset
        print("--- Task 1: Loading and Exploring the Dataset ---")
        
        # Load the Iris dataset from scikit-learn
        iris = load_iris()
        iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                               columns=iris['feature_names'] + ['species'])
        
        # Mapping the species column to human-readable names
        iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        print("\nFirst 5 rows of the dataset:")
        print(iris_df.head())
        
        print("\nData Types and Missing Values:")
        print(iris_df.info())
        
        # Check for missing values - Iris dataset is clean, but this is for demonstration
        print("\nMissing values per column:")
        print(iris_df.isnull().sum())
        
        # The Iris dataset has no missing values, so no cleaning is needed.
        # If it did, you could use:
        # iris_df.dropna() or iris_df.fillna(value)

        print("-" * 50)

        # Task 2: Basic Data Analysis
        print("--- Task 2: Basic Data Analysis ---")
        
        # Compute basic statistics
        print("\nBasic statistics of numerical columns:")
        print(iris_df.describe())
        
        # Grouping by a categorical column ('species')
        print("\nMean of numerical columns grouped by species:")
        grouped_data = iris_df.groupby('species').mean()
        print(grouped_data)
        
        print("\nInteresting Findings:")
        print("1. Setosa species generally has the smallest sepal length, sepal width, and petal length compared to the other two.")
        print("2. Virginica species has the largest average sepal length and petal length.")

        print("-" * 50)

        # Task 3: Data Visualization
        print("--- Task 3: Data Visualization ---")

        # Set up a figure for multiple plots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

        # Bar chart: Average sepal length per species
        grouped_data['sepal length (cm)'].plot(kind='bar', ax=axes[0, 0], color=['skyblue', 'salmon', 'lightgreen'])
        axes[0, 0].set_title('Average Sepal Length by Species')
        axes[0, 0].set_ylabel('Sepal Length (cm)')
        axes[0, 0].set_xlabel('Species')

        # Histogram: Distribution of petal length
        iris_df['petal length (cm)'].plot(kind='hist', ax=axes[0, 1], bins=15, color='purple', edgecolor='black')
        axes[0, 1].set_title('Distribution of Petal Length')
        axes[0, 1].set_xlabel('Petal Length (cm)')
        axes[0, 1].set_ylabel('Frequency')

        # Scatter plot: Sepal length vs. Petal length
        colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
        for species, color in colors.items():
            subset = iris_df[iris_df['species'] == species]
            axes[1, 0].scatter(subset['sepal length (cm)'], subset['petal length (cm)'], 
                               c=color, label=species, alpha=0.7)
        axes[1, 0].set_title('Sepal Length vs. Petal Length')
        axes[1, 0].set_xlabel('Sepal Length (cm)')
        axes[1, 0].set_ylabel('Petal Length (cm)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Line chart: A simple time-series plot (simulated data)
        # Note: The Iris dataset is not time-series. This is a demonstration.
        np.random.seed(42)
        simulated_data = pd.Series(np.random.randn(50).cumsum(), index=pd.date_range(start='1/1/2023', periods=50))
        simulated_data.plot(kind='line', ax=axes[1, 1], color='orange')
        axes[1, 1].set_title('Simulated Data Trend Over Time')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Value')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("Error: The specified file was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the function to execute the full assignment
if __name__ == "__main__":
    load_data_and_analyze()
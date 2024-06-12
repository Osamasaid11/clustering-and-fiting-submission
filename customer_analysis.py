import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
from scipy.stats import kurtosis, skew
import statsmodels.api as sm


# Load the dataset
data = pd.read_csv('shopping_behavior_updated.csv')

# Data Cleaning Function
def clean_data(data):
    """
    Clean and prepare the data for analysis.

    Parameters:
    data (pd.DataFrame): The dataframe containing the raw data.

    Returns:
    pd.DataFrame: Cleaned data.
    """
    # Example cleaning steps
    data.dropna(inplace=True)
    data = data[data['Age'] > 0]
    return data

def display_head(df, n=10):
    print(df.head(n))

# Function to check for missing values
def check_missing_values(df):
    print(df.isnull().sum())

# Function to check for duplicate values
def check_duplicate_values(df):
    print(df.duplicated().sum())

# Function to describe the dataset
def describe_dataset(df):
    print(df.describe().T)
# Function to extract numerical columns and calculate statistics
def calculate_statistics(df):
    numerical_df = df.select_dtypes(include=['int', 'float']).drop('Customer ID', axis=1)
    statistics = {
        'Mean': numerical_df.mean(),
        'Median': numerical_df.median(),
        'Standard Deviation': numerical_df.std(),
        'Skewness': numerical_df.apply(skew),
        'Kurtosis': numerical_df.apply(kurtosis)
    }
    statistics_df = pd.DataFrame(statistics)
    return statistics_df

# Function to plot statistics table
def plot_statistics_table(statistics_df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=statistics_df.values,
                     colLabels=statistics_df.columns,
                     rowLabels=statistics_df.index,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.savefig("statistics_table.png", bbox_inches='tight', dpi=300)
    plt.show()

# Function to plot bar chart of statistics
def plot_statistics_bar(statistics_df):
    # Reset index to turn the index into a column
    statistics_df.reset_index(inplace=True)

    # Melt the DataFrame
    melted_df = statistics_df.melt(id_vars='index', var_name='Statistic', value_name='Value')

    # Plot the statistics
    plt.figure(figsize=(10, 6))
    sns.barplot(x='index', y='Value', hue='Statistic', data=melted_df)
    plt.title('Statistics of Numeric Columns')
    plt.xlabel('Column')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.show()

# Visualizations
def plot_histogram(data):
    """
    Plot a histogram of a specified column.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data.

    Returns:
    None
    """
    plt.figure(figsize=(4, 4))
    sns.histplot(data=data, x='Age', kde=True, color='red')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()
def plot_line_chart(data):
    """
    Plot a line chart showing the relationship between seasons and average purchase amounts.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data.

    Returns:
    None
    """
    data.plot(kind='line', marker='o')
    plt.title('Average Purchase Amount by Season')
    plt.xlabel('Season')
    plt.ylabel('Average purchase amount (USD)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Function to label encode categorical columns
def encode_categorical_columns(df):
    """
    Label encode the categorical columns in the dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.

    Returns:
    pd.DataFrame: Dataframe with encoded categorical columns.
    """
    categorical_df = df.select_dtypes(include=['object', 'category'])
    label_encoder = LabelEncoder()
    encoded_df = df.copy()
    for col in categorical_df.columns:
        encoded_df[col] = label_encoder.fit_transform(df[col])
    return encoded_df

def plot_heatmap(data):
    """
    Plot a heatmap showing correlation between numerical features.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data.

    Returns:
    None
    """
    correlation_matrix = data.corr()
    plt.figure(figsize=(9, 9))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=0.2)
    plt.title('Correlation Heatmap', fontsize=25)
    plt.show()

def box_plot(data):
    """
    Create a box plot to visualize the distribution of review ratings by gender.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data.

    Returns:
    None
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Gender', y='Review Rating', data=data)
    plt.title('Review Ratings Distribution by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Review Rating')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Clustering using K-means
def kmeans_clustering(data):
    """
    Perform K-means clustering on the dataset and determine the optimal number of clusters using the elbow method.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data.

    Returns:
    None
    """
    X = data[['Age', 'Purchase Amount (USD)']]

    # Elbow Method to find the optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.show()

    # Fit K-means with the optimal number of clusters (assumed 3 for the example)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    data['Cluster'] = kmeans.fit_predict(X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='Purchase Amount (USD)', hue='Cluster', data=data, palette='viridis')
    plt.title('K-means Clustering')
    plt.xlabel('Age')
    plt.ylabel('Purchase Amount (USD)')
    plt.grid(True)
    plt.show()

# Regression Analysis
def regression_analysis(data):
    """
    Perform linear regression to investigate the relationship between Age and Purchase Amount.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data.

    Returns:
    None
    """
    X = data[['Age']]
    y = data['Purchase Amount (USD)']

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
    plt.title('Age vs. Purchase Amount (Linear Regression)')
    plt.xlabel('Age')
    plt.ylabel('Purchase Amount (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print regression details
    print('Coefficients:', model.coef_)
    print('Intercept:', model.intercept_)
    print('Mean Squared Error:', mean_squared_error(y, y_pred))

# Function to generate random indices for regplot
def generate_random_indices(encoded_df):
    return np.random.choice(np.arange(1, 3000), size=20, replace=False)

# Function to plot regression with confidence intervals
def plot_regression_with_confidence_intervals(X_feature, y_feature, random_indices):
    df_ols = pd.DataFrame({'age': X_feature.values.flatten(), 'Purchase Amount': y_feature.values.flatten()})
    df_ols['age'] = df_ols['age'].astype(float)
    X = sm.add_constant(df_ols['age'].values)
    ols_model = sm.OLS(df_ols['Purchase Amount'].values, X)
    est = ols_model.fit()
    out = est.conf_int(alpha=0.05, cols=None)
    random_df = df_ols.iloc[random_indices]
    fig, ax = plt.subplots()
    random_df.plot(x='age', y='Purchase Amount', linestyle='None', marker='s', ax=ax)
    y_pred = est.predict(X)
    x_pred = df_ols.age.values
    ax.plot(x_pred, y_pred)
    pred = est.get_prediction(X).summary_frame()
    ax.plot(x_pred, pred['mean_ci_lower'], linestyle='--', color='blue')
    ax.plot(x_pred, pred['mean_ci_upper'], linestyle='--', color='blue')

# Code Execution
if __name__ == "__main__":
    # Ensure the dataset is clean
    data = clean_data(data)

    # Statistical anaylysis
    # Calculate statistics and plot
    statistics_df = calculate_statistics(data)
    plot_statistics_table(statistics_df)
    plot_statistics_bar(statistics_df)
    
    # Generate the required visualizations
    box_plot(data)
    plot_histogram(data)

    seasonal_purchase_amount = data.groupby('Season')['Purchase Amount (USD)'].mean()
    plot_line_chart(seasonal_purchase_amount)

    # Label encode categorical columns
    encoded_df = encode_categorical_columns(data)
    plot_heatmap(encoded_df)

    # Perform clustering and regression analyses
    kmeans_clustering(data)
    regression_analysis(data)
     # Linear regression analysis
    X_feature = encoded_df[['Age']]
    y_feature = encoded_df[['Purchase Amount (USD)']]

    # Plot regression with confidence intervals
    random_indices = generate_random_indices(encoded_df)
    plot_regression_with_confidence_intervals(X_feature, y_feature, random_indices)


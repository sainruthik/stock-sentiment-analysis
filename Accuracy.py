import pandas as pd

# Load the dataset
data = pd.read_csv('C:/Users/sainr/Documents/Learning_Machine_Learning/Projects/Stock_prediction/Dataset/company_news_stories_preprocessed_with_sentiment.csv')

# Display the first few rows of the dataset
print(data.head())

# Display the column names to identify where the sentiment data might be stored
print(data.columns)

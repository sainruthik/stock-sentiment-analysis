import pandas as pd
from data_loading import load_data
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
import string

# Configure NLTK to use the correct data path
nltk.data.path.append('nltk_data')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the Punkt tokenizer manually
tokenizer = PunktSentenceTokenizer()

# Define a function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = tokenizer.tokenize(text)  # Manually tokenize the text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize the tokens
    cleaned_text = ' '.join(tokens)  # Join tokens back into a single string
    return cleaned_text

# Function to preprocess data
def preprocess_data(filepath):
    # Load the dataset using the load_data function
    data_cleaned = load_data(filepath)
    
    if data_cleaned is None:
        return None
    
    # Apply the preprocessing function to each headline column separately
    for col in data_cleaned.columns:
        if col.startswith('Top'):
            data_cleaned[col] = data_cleaned[col].apply(preprocess_text)
    
    # Display the first few rows of the cleaned text data
    print(data_cleaned.head())
    
    # Save the preprocessed data to a new CSV file
    preprocessed_filepath = filepath.replace('.csv', '_preprocessed.csv')
    data_cleaned.to_csv(preprocessed_filepath, index=False)
    
    return preprocessed_filepath

if __name__ == "__main__":
    filepath = r'Dataset\company_news_stories.csv'
    preprocess_data(filepath)

import pandas as pd
from transformers import pipeline
from data_loading import load_data

# Function to perform sentiment analysis
def perform_sentiment_analysis(filepath):
    data_cleaned = load_data(filepath)
    
    if data_cleaned is None:
        return None
    
    sentiment_analyzer = pipeline(
        'sentiment-analysis', 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )
    
    data_cleaned['story_sentiment'] = data_cleaned['story'].apply(
        lambda x: sentiment_analyzer(x)[0]['label'] if pd.notna(x) and x.strip() else 'NEUTRAL'
    )
    
    sentiment_filepath = filepath.replace('.csv', '_with_sentiment.csv')
    data_cleaned.to_csv(sentiment_filepath, index=False)
    
    return data_cleaned, sentiment_analyzer

def save_model(model, model_path):
    model.model.save_pretrained(model_path)
    model.tokenizer.save_pretrained(model_path)

def main():
    filepath = r'Dataset\company_news_stories.csv'
    preprocessed_data_filepath = filepath
    
    sentiment_data, sentiment_analyzer = perform_sentiment_analysis(preprocessed_data_filepath)
    
    model_save_path = r'Saved_Model'
    save_model(sentiment_analyzer, model_save_path)
    
    if sentiment_data is not None:
        print(sentiment_data.head())

if __name__ == "__main__":
    main()

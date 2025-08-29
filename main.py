import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import os

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

def load_csv_with_fallback(file_path):
    """
    Attempt to load CSV file with multiple encoding strategies
    Returns DataFrame if successful, None otherwise
    """
    encoding_methods = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    
    for encoding in encoding_methods:
        try:
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
            if not df.empty:
                print(f"File successfully loaded using {encoding} encoding")
                return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Unexpected error with {encoding}: {str(e)}")
            continue
    
    # Final attempt without specified encoding
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        return df
    except Exception as e:
        print(f"Failed to load file: {str(e)}")
        return None

def display_columns_with_preview(df, preview_lines=2):
    """
    Display available columns with sample data for each
    """
    print("\nAvailable columns in the CSV file:")
    print("-" * 50)
    
    for idx, column in enumerate(df.columns):
        # Get sample data from the column
        sample_data = df[column].dropna().head(preview_lines)
        sample_text = " | ".join(str(x) for x in sample_data.tolist())
        
        if len(sample_text) > 60:
            sample_text = sample_text[:57] + "..."
            
        print(f"{idx}: {column} (Sample: {sample_text})")

def select_review_column(df):
    """
    Allow user to select which column contains reviews
    """
    display_columns_with_preview(df)
    
    while True:
        try:
            choice = input("\nEnter the NUMBER of the column containing reviews: ")
            col_index = int(choice)
            
            if 0 <= col_index < len(df.columns):
                selected_column = df.columns[col_index]
                print(f"Selected column: {selected_column}")
                return selected_column
            else:
                print(f"Please enter a number between 0 and {len(df.columns)-1}")
                
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            exit()

def analyze_sentiment_batch(texts, analyzer):
    """
    Analyze sentiment for a batch of texts efficiently
    """
    sentiment_scores = []
    
    for text in texts:
        try:
            if pd.isna(text) or str(text).strip() == "":
                sentiment_scores.append(0.0)
            else:
                scores = analyzer.polarity_scores(str(text))
                sentiment_scores.append(scores['compound'])
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            sentiment_scores.append(0.0)
    
    return sentiment_scores

def main():
    """
    Main function to run the sentiment analysis workflow
    """
    # Get file path from user or use default
    file_prompt = f"Enter CSV file path ): "
    
    file_path = input(file_prompt).strip()

    
    # Verify file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    # Load CSV data
    data = load_csv_with_fallback(file_path)
    if data is None or data.empty:
        print("Failed to load data from the CSV file")
        return
    
    print(f"Successfully loaded data with {len(data)} rows and {len(data.columns)} columns")
    
    # Let user select the review column
    review_column = select_review_column(data)
    
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Perform sentiment analysis
    print("Analyzing sentiment... This may take a moment for large files.")
    review_texts = data[review_column].astype(str).tolist()
    data['sentiment_score'] = analyze_sentiment_batch(review_texts, sia)
    
    # Convert to numeric and handle any conversion issues
    data['sentiment_score'] = pd.to_numeric(data['sentiment_score'], errors='coerce')
    
    # Sort by sentiment score
    sorted_data = data.sort_values(by='sentiment_score', ascending=False)
    
    # Get top and bottom reviews
    top_positive = sorted_data.head(40)
    top_negative = sorted_data.tail(40)
    
    # Display results
    print("\n" + "="*60)
    print("TOP 40 MOST POSITIVE REVIEWS:")
    print("="*60)
    for idx, (_, row) in enumerate(top_positive.iterrows(), 1):
        preview = str(row[review_column])[:100] + "..." if len(str(row[review_column])) > 100 else str(row[review_column])
        print(f"{idx}. Score: {row['sentiment_score']:.3f} - {preview}")
    
    print("\n" + "="*60)
    print("TOP 40 MOST NEGATIVE REVIEWS:")
    print("="*60)
    for idx, (_, row) in enumerate(top_negative.iterrows(), 1):
        preview = str(row[review_column])[:100] + "..." if len(str(row[review_column])) > 100 else str(row[review_column])
        print(f"{idx}. Score: {row['sentiment_score']:.3f} - {preview}")
    

if __name__ == "__main__":
    main()

# Sentiment Analysis

A Python tool that reads a CSV file of text, such as product reviews, game reviews or comments, scores the sentiment of every entry, and shows the **40 most positive** and **40 most negative** entries.

Sentiment is scored with **VADER** from the Natural Language Toolkit (NLTK). VADER is designed for short, informal text like reviews and social media posts.

## Features

- **Works with messy CSV files:** detects the file's text encoding automatically and handles different CSV formats.
- **Choose your column:** previews the data so you can pick which column holds the text to analyse.
- **Batch scoring:** scores every row with VADER's compound score, from -1 (most negative) to +1 (most positive).
- **Ranked results:** displays the top 40 positive and top 40 negative entries with their scores.
- **Error handling:** gives clear messages for missing files, encoding problems and badly formatted data, instead of crashing.

## Requirements

- Python 3.8 or newer
- [pandas](https://pandas.pydata.org/)
- [NLTK](https://www.nltk.org/) with the VADER lexicon

## Installation

```bash
git clone https://github.com/marcus2045/Sentiment-Analysis.git
cd Sentiment-Analysis
pip install pandas nltk
python -m nltk.downloader vader_lexicon
```

## Usage

```bash
python main.py
```

1. Enter the path to your CSV file when asked.
2. Check the data preview and choose the column that contains the text.
3. The tool scores every entry and prints the 40 most positive and 40 most negative results.

## Test data

The tool was tested with two public review datasets from Kaggle:

- Steam game reviews
- Amazon product reviews

Download either dataset as a CSV file and run the tool on it.

## How the scoring works

VADER gives each piece of text a **compound score** between -1 and +1:

| Compound score | Meaning |
| --- | --- |
| 0.05 or higher | Positive |
| Between -0.05 and 0.05 | Neutral |
| -0.05 or lower | Negative |

It accounts for things like capital letters ("GREAT"), exclamation marks, emojis and words like "not" or "very", which makes it well suited to reviews.

## Possible improvements

- Save the results to a new CSV file
- Add a chart showing how positive and negative the whole dataset is
- Add a simple graphical interface

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

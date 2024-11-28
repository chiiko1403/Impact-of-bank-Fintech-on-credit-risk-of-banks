from google.colab import drive
drive.mount('/content/gdrive', force_remount = True)

main_folder_path = '/content/gdrive/My Drive/KLTN/Preprocessing'

!pip install PyPDF2
!pip install vncorenlp
!pip install underthesea
!pip install nltk

# Import necessary libraries
import re
import string
from collections import Counter
import os
from underthesea import text_normalize
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Specify the path to the Vietnamese stopword file
stopword_file_path = '/content/gdrive/My Drive/KLTN/vietnamese-stopwords.txt'  # Replace with your file path

# Load stopwords from the file
with open(stopword_file_path, 'r', encoding='utf-8') as f:
    vietnamese_stopwords = set([text_normalize(line.strip().lower()) for line in f.readlines() if line.strip()])

# Print a sample of the stopwords to confirm they're loaded correctly
print("Loaded stopwords:", list(vietnamese_stopwords)[:10])

def preprocess_text(text):
    # Normalize text for Vietnamese (diacritics, phonetics)
    text = text_normalize(text)
    # Convert text to lowercase
    text = text.lower()
    # Replace escape sequences with a space
    text = re.sub(r'\\[nrtbf]', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits (numbers)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove Vietnamese stopwords
    text = ' '.join([word for word in text.split() if word not in vietnamese_stopwords])
    return text

# Path to the FinTech keywords file
fintech_keyword_file_path = '/content/gdrive/My Drive/KLTN/fintech-keywords.txt'

# Load the FinTech keywords from the file
with open(fintech_keyword_file_path, 'r', encoding='utf-8') as f:
    fintech_keywords = [text_normalize(line.strip().lower()) for line in f.readlines() if line.strip()]

# Print a sample of the loaded keywords to verify
print("Loaded FinTech keywords:", fintech_keywords[:10])

import os
import re
import pandas as pd

# Dictionary to store keyword frequencies per bank per year
keyword_frequencies_per_bank_year = {}

# Iterate through each subfolder (bank)
for bank_folder in os.listdir(main_folder_path):
    bank_folder_path = os.path.join(main_folder_path, bank_folder)

    if os.path.isdir(bank_folder_path):  # Only proceed if it's a directory (subfolder)
        # Initialize a nested dictionary to hold the keyword frequencies for each year for the current bank
        bank_keyword_frequencies = {}

        # Iterate through each .txt file in the bank's folder
        for file_name in os.listdir(bank_folder_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(bank_folder_path, file_name)

                # Extract the year from the filename using a regular expression (assuming the year is in the filename)
                year_match = re.search(r'\d{4}', file_name)
                if year_match:
                    year = year_match.group(0)  # Extract the year as a string
                else:
                    year = 'Unknown'  # Handle cases where no year is found

                # Read and preprocess the text
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    cleaned_text = preprocess_text(text)  # Preprocess the text

                # Calculate total words
                total_words = len(cleaned_text.split())

                # Calculate keyword frequencies for this report (as frequency instead of occurrences)
                keyword_frequencies = {keyword: cleaned_text.count(keyword) / total_words if total_words > 0 else 0 for keyword in fintech_keywords}

                # Aggregate frequencies for this bank in the corresponding year
                if year not in bank_keyword_frequencies:
                    bank_keyword_frequencies[year] = {'total_words': total_words, 'total_keyword_frequency': 0}  # Initialize total_words and total_keyword_frequency

                # Update total keyword frequency
                bank_keyword_frequencies[year]['total_keyword_frequency'] += sum(keyword_frequencies.values())

                # Update keyword frequencies for the year
                for keyword, freq in keyword_frequencies.items():
                    if keyword in bank_keyword_frequencies[year]:
                        bank_keyword_frequencies[year][keyword] += freq
                    else:
                        bank_keyword_frequencies[year][keyword] = freq

        # Store the aggregated keyword frequencies for the current bank
        keyword_frequencies_per_bank_year[bank_folder] = bank_keyword_frequencies

# Prepare the data for the DataFrame
data = []
for bank, yearly_data in keyword_frequencies_per_bank_year.items():
    for year, keyword_frequencies in yearly_data.items():
        row = {'Bank': bank, 'Year': year}
        total_words = keyword_frequencies['total_words']  # Retrieve the total word count
        total_keyword_frequency = keyword_frequencies['total_keyword_frequency']  # Retrieve the total keyword frequency

        # Add the total keyword frequency to the row
        row['Total_Keyword_Frequency'] = total_keyword_frequency

        # Add individual keyword frequencies to the row
        for keyword, freq in keyword_frequencies.items():
            if keyword not in ['total_words', 'total_keyword_frequency']:  # Exclude keys we don't want in the final output
                row[keyword] = freq

        data.append(row)

# Create a DataFrame
df_keyword_frequencies = pd.DataFrame(data)

# Display the DataFrame
df_keyword_frequencies.head()

# Optionally, save the DataFrame to a CSV file
df_keyword_frequencies.to_csv('/content/gdrive/My Drive/KLTN/keyword_frequencies.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/content/gdrive/My Drive/KLTN/keyword_frequencies.csv')

# Extract only the keyword columns (assumed from column index 2 onwards)
keyword_columns = df.columns[3:]

# Apply Min-Max normalization to keyword columns
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[keyword_columns] = scaler.fit_transform(df[keyword_columns])

# Calculate the proportion for each keyword by dividing each element by the column sum
df_proportions = df_normalized[keyword_columns].div(df_normalized[keyword_columns].sum(axis=0), axis=1)

# Calculate entropy for each keyword
entropy = -np.nansum(df_proportions * np.log(df_proportions + 1e-9), axis=0) / np.log(len(df))
entropy_df = pd.DataFrame({
    'Keyword': keyword_columns,
    'Entropy': entropy
})

# Calculate the weight for each keyword
weights = (1 - entropy) / (1 - entropy).sum()

# Display the weights
weights_df = pd.DataFrame({
    'Keyword': keyword_columns,
    'Weight': weights
})

# Calculate the fintech index for each bank-year
df_normalized['Fintech_Index'] = np.dot(df_normalized[keyword_columns], weights)

# Save the DataFrame to a CSV file
df_normalized.to_csv('/content/gdrive/My Drive/KLTN/fintech_index_output.csv', index=False)
import pandas as pd
from KeyMojiAPI import KeyMoji

# Load the uploaded file
file_path = '/Users/czu/Desktop/VSC/DataScience[NTU1131]/Project_BubbleTea/dummy.csv'
data = pd.read_csv(file_path)

# Display the first few rows to understand its structure
# print(data.head())

# Initialize a list to store the calculated scores for each row in the 'text' column
scores_list = []

# Placeholder example of how to process each text entry
# This code assumes that keymoji.sense2() works as shown in the user's example
# Since I cannot run the KeyMojiAPI directly here, I'll create a mock function to simulate it
keymoji = KeyMoji(username="seizuhsu2012@gmail.com", keymojiKey="9jEj6cXB@_GC*nuIowUMQpiPfsLxByW")

# Process each text entry in the 'text' column using the actual sense2 API call
for text in data['text']:
    # Call the KeyMoji sense2 API
    sense2Result = keymoji.sense2(text, model="general", userDefinedDICT={"positive":[], "negative":[], "cursing":[]})
    
    if sense2Result.get('status') and 'results' in sense2Result:
        # Extract scores
        scores = [result['score'] for result in sense2Result['results']]
        # Calculate the overall score
        overall_score = sum(scores) / len(scores) if scores else 0
        scaled_score = (overall_score + 1) * 2.5
    else:
        # Handle cases where there is no 'results' key or status is False
        overall_score = 0  # Default score in case of an error or missing data
    scores_list.append(scaled_score)

# print(scores_list)

# Add the scores as a new column in the DataFrame
data['sentiment_score'] = scores_list
data.to_csv('trysentimentscore.csv')

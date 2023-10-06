#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import the libraries
import json
import pandas as pd
import openai
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import requests
from gettext import install
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import datetime
import math


print("\n\n******* Python script started *******\n\n")

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
# pythonBaseUrl = sys.argv[1]
# webhookUrl = sys.argv[2]
webhookUrl = ""
# print("pythonBaseUrl : ", pythonBaseUrl)

# Classification using Random Forest


def convert(date_time):
    format = "%Y-%m-%d"

    datetime_str = datetime.datetime.strptime(date_time, format)

    return datetime_str


def classify_posts(posts, accuracy=0.7, window_days=10):
    print("callling classify_posts")
    # Convert accuracy to similarity threshold (0 <= similarity <= 1)
    similarity_threshold = accuracy
    # Use TF-IDF Vectorizer to convert text to vectors
    tfidf = TfidfVectorizer().fit_transform(posts["message"])

    # Create a classification column initialized with "U"
    posts["classification"] = "U"

    for i, post in posts.iterrows():
        # Compare with posts within the window period
        start_date = post["date"] - datetime.timedelta(days=window_days)
        end_date = post["date"] + datetime.timedelta(days=window_days)
        window_posts = posts[
            (posts["date"] >= start_date) & (posts["date"] <= end_date)
        ]

        for j, compare_post in window_posts.iterrows():
            # Skip comparison with self
            if i == j:
                continue

            # Calculate similarity
            similarity = linear_kernel(tfidf[i], tfidf[j]).flatten()[0]

            # If similarity exceeds threshold and compare_post date is before post date
            if (
                similarity > similarity_threshold
                and compare_post["date"] < post["date"]
            ):
                posts.at[i, "classification"] = "R"
                break

    return posts


def classifier_rf():
    print("callling classifier_rf")
    # Load the main data and remove any null values present
    # Open the JSON file with the correct encoding
    with open(
        "10k Master.json",
        encoding="utf-8",
    ) as f:
        # Read the contents of the file
        contents = f.read()

    # Replace invalid control characters with spaces
    cleaned_contents = "".join(char if ord(char) < 128 else " " for char in contents)

    # Parse the cleaned JSON data
    main_data = json.loads(cleaned_contents, strict=False)
    main_data = pd.DataFrame(main_data)
    main_data = main_data[pd.notnull(main_data["Vernon Sub Theme"])]
    main_data = main_data[pd.notnull(main_data["Vernon Main Theme"])]
    main_data = main_data[pd.notnull(main_data["Vernon Sub Sub Theme"])]
    main_data = main_data[pd.notnull(main_data["Message"])]

    # Create the tf-idf vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the messages in the main data
    tfidf_vectorizer.fit(main_data["Message"])

    # Transform the messages in the main data using the fitted vectorizer
    main_data_tfidf = tfidf_vectorizer.transform(main_data["Message"])

    # Train a Random Forest classifier
    num_trees = 400
    rf = RandomForestClassifier(n_estimators=num_trees)

    # Fit the classifier on the themes and subthemes in the main data
    rf.fit(
        main_data_tfidf,
        main_data["Vernon Main Theme"]
        + "-"
        + main_data["Vernon Sub Theme"]
        + "-"
        + main_data["Vernon Sub Sub Theme"],
    )
    # payloadJson = json.loads(sys.argv[3])
    payloadJson = [
        {
            "id": "651662f26d85af737052ffc8",
            "message": '"Be sure that there is plenty of space for lawns and gardens. Reserve large areas for football, hockey and parks." â€”Jamsetji Tata As we near the eve of our Founder\'s 184th birth anniversary, we celebrate his vision of both industry &amp; community. #LegendLivesOn (1/2) pic.twitter.com/Gc5YJ5YXkb',
            "Company Name": "tata group",
            "Publish Date/Time": "2023-03-01 18:42:28",
        },
        {
            "id": "651662f26d85af737052ffce",
            "message": '"Be sure that there is plenty of space for lawns and gardens. Reserve large areas for football, hockey and parks." â€”Jamsetji Tata More than just an industrial beacon, Jamshedpur was the epitome of our Founder\'s vision and values. As we near the eve of his 184th birth anniversary, we celebrate his pioneering spirit. Want to know more about the 1st planned industrial city? Click the link in bio. #LegendLivesOn #ThisIsTata',
            "Company Name": "tata group",
            "Publish Date/Time": "2023-03-01 18:42:28",
        },
        {
            "id": "651662f36d85af737052ffd4",
            "message": '"Be sure that there is plenty of space for lawns and gardens. Reserve large areas for football, hockey and parks." â€”Jamsetji Tata More than just an industrial beacon, Jamshedpur was the epitome of our Founder\'s vision and values. As we near the eve of his 184th birth anniversary, we celebrate his pioneering spirit. Want to know more about the 1st planned industrial city? Click here: https://www.youtube.com/watch?v=TgOLdE6cr40 #LegendLivesOn #ThisIsTata',
            "Company Name": "tata group",
            "Publish Date/Time": "2023-03-01 18:42:28",
        },
    ]
    # print("payloadJson")
    # print(payloadJson)

    # Prepare the new data from the provided list of dictionaries
    new_data = pd.DataFrame(
        [
            {
                "id": entry["id"],
                "message": entry["message"],
                "company": entry["Company Name"],
                "date": entry["Publish Date/Time"],
            }
            for entry in payloadJson
        ]
    )

    # print("new_data")
    # print(new_data)
    # print("new_data 2 \n\n")
    # print(new_data["message"])

    # Remove duplicates from the new data DataFrame based on the 'message' column
    new_data = new_data.drop_duplicates(subset=["message"])

    # Transform the messages in the new data using the fitted vectorizer
    new_data_tfidf = tfidf_vectorizer.transform(new_data["message"])

    # Classify the themes and subthemes in the new data
    predicted_probs = rf.predict_proba(new_data_tfidf)

    # Extract the two highest probabilities and their associated themes, subthemes, and sub-subthemes for each row
    themes_probs = []
    subthemes_probs = []
    subsubthemes_probs = []
    for probs in predicted_probs:
        themes_indices = np.argsort(probs)[-3:]
        themes_probs.append(
            [(rf.classes_[i].split("-")[0], probs[i]) for i in themes_indices]
        )
        subthemes_indices = np.argsort(probs)[-3:]
        subthemes_probs.append(
            [(rf.classes_[i].split("-")[1], probs[i]) for i in subthemes_indices]
        )
        subsubthemes_indices = np.argsort(probs)[-3:]
        subsubthemes_probs.append(
            [(rf.classes_[i].split("-")[2], probs[i]) for i in subsubthemes_indices]
        )

    # Assign the themes, subthemes, and sub-subthemes to the new data DataFrame
    new_data["Theme_1"] = [t[0][0] for t in themes_probs]
    new_data["ptheme_1"] = [t[0][1] for t in themes_probs]
    new_data["Subtheme1"] = [t[0][0] for t in subthemes_probs]
    new_data["psubtheme1"] = [t[0][1] for t in subthemes_probs]
    new_data["Subsubtheme1"] = [t[0][0] for t in subsubthemes_probs]
    new_data["psubsubtheme1"] = [t[0][1] for t in subsubthemes_probs]
    new_data["Theme_2"] = [t[1][0] for t in themes_probs]
    new_data["ptheme_2"] = [t[1][1] for t in themes_probs]
    new_data["Subtheme2"] = [t[1][0] for t in subthemes_probs]
    new_data["psubtheme2"] = [t[1][1] for t in subthemes_probs]
    new_data["Subsubtheme2"] = [t[1][0] for t in subsubthemes_probs]
    new_data["psubsubtheme2"] = [t[1][1] for t in subsubthemes_probs]
    new_data["Theme_3"] = [t[2][0] for t in themes_probs]
    new_data["ptheme_3"] = [t[2][1] for t in themes_probs]
    new_data["Subtheme3"] = [t[2][0] for t in subthemes_probs]
    new_data["psubtheme3"] = [t[2][1] for t in subthemes_probs]
    new_data["Subsubtheme3"] = [t[2][0] for t in subsubthemes_probs]
    new_data["psubsubtheme3"] = [t[2][1] for t in subsubthemes_probs]
    return new_data


# Print the first 15 rows of the new data DataFrame
new_data = classifier_rf()
print("calculated new_data success")

# Mark duplicates as 'Duplicate' and unique as 'Unique'
# Specify the column you want to check for unique/duplicate values
column_to_check = "message"  # Replace with your column name

# Set the predefined match percentage threshold
similarity_threshold = 0.8

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the messages into TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(new_data[column_to_check])

# Calculate the cosine similarity between all message pairs
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Initialize the 'Tag' column with "Unique" for all records
new_data['Tag'] = "Unique"

# Iterate through the similarity matrix to identify duplicates
for i in range(len(new_data)):
    if new_data.at[i, 'Tag'] == "Duplicate":
        continue
    
    for j in range(i+1, len(new_data)):
        if new_data.at[j, 'Tag'] == "Duplicate":
            continue
        
        similarity_score = cosine_sim[i][j]
        
        if similarity_score >= similarity_threshold:
            # Mark both messages as "Duplicate"
            new_data.at[i, 'Tag'] = "Duplicate"
            new_data.at[j, 'Tag'] = "Duplicate"


# Extraction of Keywords and Keyphrases
data = new_data

print("calculated duplicate_status success")


def extract_keywords(data):
    print("calling extract_keywords")
    # Load the dataset
    data = new_data

    # Define a function to extract keywords and keyphrases from a message
    def extract_keywords(message):
        # Tokenize the message into words
        words = word_tokenize(message.lower())
        # Remove stop words
        stop_words = set(stopwords.words("english"))
        words = [word for word in words if word not in stop_words]
        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        # Tag the words with their part-of-speech
        tagged_words = pos_tag(words)
        # Extract nouns, adjectives, and verbs
        keywords = [
            word
            for word, tag in tagged_words
            if tag.startswith("N") or tag.startswith("J") or tag.startswith("V")
        ]
        # Combine adjacent words into keyphrases
        keyphrases = []
        i = 0
        while i < len(keywords):
            j = i + 1
            while j < len(keywords) and tagged_words[j][1].startswith(("N", "J", "V")):
                j += 1
            keyphrase = " ".join(keywords[i:j])
            if len(keyphrase.split()) > 1:
                keyphrases.append(keyphrase)
            i = j
        return keywords, keyphrases

    # Apply the function to each message in the dataset
    data["Keywords"], data["Keyphrases"] = zip(*data["message"].apply(extract_keywords))

    return data


data = extract_keywords(data)

print("calculated extract_keywords success")
# Named Entity Recognition (NER)


def extract_entities(data):
    print("callling extract_entities")
    # Set up the OpenAI API credentials
    openai.api_key = "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"

    # Define the entity types to extract
    entity_types = [
        "Person Names",
        "Organization",
        "Hash Tags",
        "Location",
        "Brand",
        "Category",
        "URLs",
    ]

    # Define a function to extract entities from a single message
    def extract_entities(message):
        # Use the OpenAI API to perform zero-shot NER on the message
        prompt = f"Please extract the following entity types from the text: {', '.join(entity_types)}. \n\nText: {message}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        # Extract the entity labels and scores from the API response
        entities = []
        if response["choices"]:
            entities = response["choices"][0]["text"].split("\n")
        entity_dict = {et: None for et in entity_types}
        for entity in entities:
            for et in entity_types:
                if et in entity:
                    entity_parts = entity.split(":")
                    if len(entity_parts) > 1:
                        entity_dict[et] = entity_parts[1].strip()
        # Return the extracted entities as a dictionary
        return entity_dict

    # Load the data into a pandas DataFrame
    df = data

    # Extract entities from each message in the DataFrame
    entities = df["message"].apply(extract_entities).tolist()

    # Convert the list of entity dictionaries into a DataFrame
    entity_df = pd.DataFrame(entities, columns=entity_types)

    # Concatenate the original DataFrame with the entity DataFrame
    result_df = pd.concat([df, entity_df], axis=1)

    return result_df


result_df = extract_entities(data)

print("calculated extract_entities success")
# Summarization and Gists


def extract_entities_and_generate_summary_gist(data):
    print("\n extract_entities_and_generate_summary_gist => calculating")
    # Set up the OpenAI API credentials
    openai.api_key = "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"

    # Define the entity types to extract
    entity_types = [
        "Person Names",
        "Organisation",
        "Hash Tags",
        "location",
        "brands",
        "category",
        "URLs",
    ]

    # Define a function to extract entities from a single message
    def generate_summary_and_gist(message):
        # Use the OpenAI API to perform zero-shot NER on the message
        prompt = f'Summarize this text and skip any text which has less than 3 words or is a one sentence: "{message}" in 2-3 sentences.'
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301", messages=[{"role": "user", "content": prompt}]
        )

        # Extract the entity labels and scores from the API response
        summary = response["choices"][0]["message"]["content"]

        # Use the OpenAI API to perform zero-shot NER on the message
        prompt_gist = f'Generate a gist for this text and skip any text which has less than 3 words or is a one sentence: "{message}" in 1 sentence.'
        response_gist = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[{"role": "user", "content": prompt_gist}],
        )

        # Extract the entity labels and scores from the API response
        gist = response_gist["choices"][0]["message"]["content"]

        return summary, gist

    # Load the data into a pandas DataFrame
    df = data

    # Extract entities from each message in the DataFrame
    entities = df["message"].apply(lambda x: {et: None for et in entity_types}).tolist()

    # Convert the list of entity dictionaries into a DataFrame
    entity_df = pd.DataFrame(entities, columns=entity_types)

    # Concatenate the original DataFrame with the entity DataFrame
    result_df = pd.concat([df, entity_df], axis=1)

    # Generate summaries and gists for each message in the DataFrame
    summaries = []
    gists = []
    for text in result_df["message"]:
        summary, gist = generate_summary_and_gist(text)
        summaries.append(summary)
        gists.append(gist)

    # Save the summaries and gists in the dataframe
    result_df["Summary"] = summaries
    result_df["Gist"] = gists

    return result_df


# Resultant Data Frame
df = extract_entities_and_generate_summary_gist(result_df)
print("calculated extract_entities_and_generate_summary_gist success")


# Replace NaN values with a default value (e.g., 0)
df = df.fillna(0)

payloadData = df.head(len(df)).to_dict()

url = webhookUrl

print(url)

payload = json.dumps({"data": payloadData})
headers = {"Content-Type": "application/json"}

print("\n\n******* Webhook calling *******\n\n")
print(payload)

# response = requests.request("POST", url, headers=headers, data=payload)

# print(response.text)
print("\n\n******* Python script ended *******\n\n")

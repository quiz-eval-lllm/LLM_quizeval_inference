import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import FastText
from collections import Counter
from scipy.spatial.distance import euclidean
import numpy as np
import joblib
import pickle
import os
import logging

# TODO: Refactoring resource when inference manager start
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# TEMPORARY FUNCTION


async def calculate_essay_score(user_ans, expected_ans):
    """
    Calculate scores for a list of user answers compared to expected answers
    using heuristic-based scoring without a pre-trained model.

    Parameters:
        user_ans (list of str): User's answers.
        expected_ans (list of str): Expected answers.

    Returns:
        tuple: A tuple containing a list of individual scores and the total score (both as floats).
    """

    # Functions for preprocessing and scoring
    def preprocess_text(text):
        ps = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum()
                  and word not in stop_words]
        tokens = [ps.stem(word) for word in tokens]
        return ' '.join(tokens)

    def jaccard_distance(s1, s2):
        set1 = set(s1.split())
        set2 = set(s2.split())
        return 1 - (len(set1.intersection(set2)) / len(set1.union(set2)))

    def calculate_euclidean_distance(vector1, vector2):
        return euclidean(vector1, vector2)

    # Preprocess the data
    processed_user_ans = [preprocess_text(ans) for ans in user_ans]
    processed_expected_ans = [preprocess_text(ans) for ans in expected_ans]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    vectorizer.fit(processed_user_ans + processed_expected_ans)

    # Calculate scores using similarity measures
    scores = []
    for user, expected in zip(processed_user_ans, processed_expected_ans):
        user_vec = vectorizer.transform([user]).toarray()[0]
        expected_vec = vectorizer.transform([expected]).toarray()[0]

        # Cosine similarity
        cosine_sim = cosine_similarity([user_vec], [expected_vec])[0][0]

        # Jaccard distance
        jac_dist = jaccard_distance(user, expected)

        # Euclidean distance
        euclidean_dist = calculate_euclidean_distance(user_vec, expected_vec)

        # Calculate a heuristic score (weighted sum of metrics)
        score = (cosine_sim * 0.5) + ((1 - jac_dist) * 0.3) + \
            (1 / (1 + euclidean_dist) * 0.2)

        # Normalize score to 0-100 range
        normalized_score = round(score * 100, 2)
        scores.append(normalized_score)

    # Calculate total score
    total_score = round(sum(scores), 2)

    return scores, total_score

# async def calculate_essay_score(user_ans, expected_ans):
#     """
#     Calculate scores for a list of user answers compared to expected answers.

#     Parameters:
#         user_ans (list of str): User's answers.
#         expected_ans (list of str): Expected answers.
#         model_name (str): Path to the trained regression model file.

#     Returns:
#         dict: A dictionary containing `final_score` (float) and `score` (list of float).
#     """

#     # Functions for preprocessing and scoring

#     def preprocess_text(text):
#         ps = PorterStemmer()
#         stop_words = set(stopwords.words('english'))
#         text = text.lower()
#         tokens = word_tokenize(text)
#         tokens = [word for word in tokens if word.isalnum()
#                   and word not in stop_words]
#         tokens = [ps.stem(word) for word in tokens]
#         return ' '.join(tokens)

#     def jaccard_distance(s1, s2):
#         set1 = set(s1.split())
#         set2 = set(s2.split())
#         return 1 - (len(set1.intersection(set2)) / len(set1.union(set2)))

#     def sentence_length(s):
#         return len(s.split())

#     def unique_overlap(s1, s2):
#         set1 = set(s1.split())
#         set2 = set(s2.split())
#         return len(set1.intersection(set2))

#     def calculate_euclidean_distance(vector1, vector2):
#         return euclidean(vector1, vector2)

#     # Preprocess the data
#     processed_user_ans = [preprocess_text(ans) for ans in user_ans]
#     processed_expected_ans = [preprocess_text(ans) for ans in expected_ans]

#     # TF-IDF Vectorization
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(
#         processed_user_ans + processed_expected_ans)

#     results = []
#     for user, expected in zip(processed_user_ans, processed_expected_ans):
#         user_vec = vectorizer.transform([user]).toarray()[0]
#         expected_vec = vectorizer.transform([expected]).toarray()[0]

#         cosine_sim = cosine_similarity([user_vec], [expected_vec])[0][0]
#         jac_dist = jaccard_distance(user, expected)
#         length_user = sentence_length(user)
#         length_expected = sentence_length(expected)
#         overlap = unique_overlap(user, expected)
#         euclidean_dist = calculate_euclidean_distance(user_vec, expected_vec)

#         results.append({
#             "cosine_similarity": cosine_sim,
#             "jaccard_distance": jac_dist,
#             "length_user": length_user,
#             "length_expected": length_expected,
#             "unique_overlap": overlap,
#             "euclidean_distance": euclidean_dist
#         })

#     # Load the trained voting regressor model
#     model_name = os.path.abspath(os.path.join(os.path.dirname(
#         __file__), '../models/voting_regressor_model.pkl'))
#     try:
#         loaded_model = joblib.load(model_name)
#     except AttributeError as e:
#         print(f"Warning: Attribute mismatch in the model file - {e}")

#     expected_features = ["cosine_similarity",
#                          "jaccard_distance", "unique_overlap", "euclidean_distance"]
#     if not all(f in feature_names for f in expected_features):
#         raise ValueError("Mismatch in expected features for the model.")

#     # Check if the model supports prediction
#     if not hasattr(loaded_model, 'predict'):
#         raise ValueError("The loaded object is not a valid model.")

#     # Predict scores for each result
#     feature_names = ["cosine_similarity", "jaccard_distance",
#                      "unique_overlap", "euclidean_distance"]
#     scores = []

#     for result in results:
#         try:
#             row_features = pd.DataFrame([[
#                 result["cosine_similarity"],
#                 result["jaccard_distance"],
#                 result["unique_overlap"],
#                 result["euclidean_distance"]
#             ]], columns=feature_names)
#             predicted_value = loaded_model.predict(row_features)
#             formatted_score = round(predicted_value[0] * 100, 2)
#             scores.append(formatted_score)
#         except Exception as e:
#             logging.info(f"Prediction failed for features {result}: {e}")
#             scores.append(0.0)

#     return scores, sum(scores)

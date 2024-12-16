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


async def calculate_essay_score(user_ans, expected_ans):

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

    def sentence_length(s):
        return len(s.split())

    def unique_overlap(s1, s2):
        set1 = set(s1.split())
        set2 = set(s2.split())
        return len(set1.intersection(set2))

    def calculate_euclidean_distance(vector1, vector2):
        return euclidean(vector1, vector2)

    # Preprocess the data
    logging.info("[1] Word Vectorizing")
    processed_user_ans = [preprocess_text(ans) for ans in user_ans]
    processed_expected_ans = [preprocess_text(ans) for ans in expected_ans]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(
        processed_user_ans + processed_expected_ans)

    results = []
    logging.info("[2] Calculating Features")
    for user, expected in zip(processed_user_ans, processed_expected_ans):
        user_vec = vectorizer.transform([user]).toarray()[0]
        expected_vec = vectorizer.transform([expected]).toarray()[0]

        cosine_sim = cosine_similarity([user_vec], [expected_vec])[0][0]
        jac_dist = jaccard_distance(user, expected)
        length_user = sentence_length(user)
        length_expected = sentence_length(expected)
        overlap = unique_overlap(user, expected)
        euclidean_dist = calculate_euclidean_distance(user_vec, expected_vec)

        results.append({
            "cosine_similarity_tfidf": cosine_sim,
            "jaccard_distance": jac_dist,
            "length_user": length_user,
            "length_expected": length_expected,
            "unique_overlap": overlap,
            "euclidean_distance_tfidf": euclidean_dist
        })

    # Load the trained voting regressor model
    model_name = os.path.abspath(os.path.join(os.path.dirname(
        __file__), '../models/eval_model.pkl'))
    try:
        loaded_model = joblib.load(model_name)
    except AttributeError as e:
        print(f"Warning: Attribute mismatch in the model file - {e}")

    feature_names = ["cosine_similarity_tfidf", "jaccard_distance",
                     "unique_overlap", "euclidean_distance_tfidf"]

    expected_features = ["cosine_similarity_tfidf",
                         "jaccard_distance", "unique_overlap", "euclidean_distance_tfidf"]
    if not all(f in feature_names for f in expected_features):
        raise ValueError("Mismatch in expected features for the model.")

    # Check if the model supports prediction
    if not hasattr(loaded_model, 'predict'):
        raise ValueError("The loaded object is not a valid model.")

    scores = []

    logging.info("[3] Calculating Score")
    for result in results:
        try:
            row_features = pd.DataFrame([[
                result["cosine_similarity_tfidf"],
                result["jaccard_distance"],
                result["unique_overlap"],
                result["euclidean_distance_tfidf"]
            ]], columns=feature_names)
            predicted_value = loaded_model.predict(row_features)
            formatted_score = round(predicted_value[0] * 10, 2)
            scores.append(formatted_score)
        except Exception as e:
            logging.info(f"Prediction failed for features {result}: {e}")
            scores.append(0.0)

    return scores, sum(scores)

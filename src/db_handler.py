import asyncio
import os
import pandas as pd
import logging
from dotenv import load_dotenv
from utils.db_utils import DatabaseUtility

load_dotenv()

# Database connection configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT"))
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA")

# Initialize the database utility
db_util = DatabaseUtility(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    db=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    schema=POSTGRES_SCHEMA)

# TODO: fetch package

# TODO: update question


async def fetch_evaluation(evaluation_id):
    """
    Fetch evaluation details by evaluation_id.
    """
    query = f"SELECT * FROM {POSTGRES_SCHEMA}.evaluation WHERE eval_id = $1"
    try:
        data = await db_util.fetch_data(query, evaluation_id)
        return data if data else []  # Return a list of dictionaries
    except Exception as e:
        logging.error(f"Error fetching evaluation: {e}")
        return []


async def fetch_question(question_id):
    """
    Fetch queation details by question_id.
    """
    query = f"SELECT * FROM {POSTGRES_SCHEMA}.quiz_essay WHERE essay_id = $1"

    try:
        data = await db_util.fetch_data(query, question_id)
        return pd.DataFrame(data) if data else pd.DataFrame

    except Exception as e:
        logging.info(f"Error fetching evaluation: {e}")
        return pd.DataFrame()


async def update_evaluation(evaluation_id, score):
    """
    Update score by evaluation id.
    """
    query = f"UPDATE {POSTGRES_SCHEMA}.evaluation SET score = $1 WHERE eval_id = $2 RETURNING id"
    try:
        result = await db_util.post_data(query, score, evaluation_id)
        return {"status": "success", "message": "Evaluation updated successfully"} if result else {"status": "error", "message": "Evaluation not found"}
    except Exception as e:
        print(f"Error updating evaluation: {e}")
        return {"status": "error", "message": str(e)}


async def update_quiz_activites(quiz_id, final_score):
    """
    Update final score by quiz id.
    """
    query = f"UPDATE {POSTGRES_SCHEMA}.quiz_activities SET final_score = $1 WHERE quiz_id = $2 RETURNING id"
    try:
        result = await db_util.post_data(query, final_score, quiz_id)
        return {"status": "success", "message": "Quiz activity updated successfully"} if result else {"status": "error", "message": "Quiz activity not found"}
    except Exception as e:
        print(f"Error updating quiz activity: {e}")
        return {"status": "error", "message": str(e)}

import asyncio
import tempfile
import os
import uuid
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


async def fetch_package(package_id):
    """
    Fetch package details by package_id, focusing on binary data (context) for PDF processing.
    """
    query = f"SELECT * FROM {POSTGRES_SCHEMA}.package WHERE package_id = $1"
    try:
        data = await db_util.fetch_data(query, package_id)
        if not data:
            return None

        record = data[0]
        package = dict(record)

        if "context" in package and package["context"]:
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_pdf.write(package["context"])
            temp_pdf.close()
            package["pdf_path"] = temp_pdf.name

        return package

    except Exception as e:
        logging.error(f"Error fetching package: {e}")
        return None


async def insert_essay_questions(package_id, questions, answers):
    """
    Insert the series of essay questions
    """
    query = f"INSERT INTO {POSTGRES_SCHEMA}.quiz_essay(essay_id, package, question, context, is_deleted)
    VALUES($1, $2, $3, $4, $5) RETURNING essay_id"

    if len(questions) != len(answers):
        return {"status": "error", "message": "Questions and answers must have the same length"}

    try:
        inserted_ids = []
        for (question, answer) in enumerate(zip(questions, answers)):
            essay_id = str(uuid.uuid4())
            result = await db_util.post_data(
                query,
                essay_id,
                package_id,
                question,
                answer,
                False
            )
            if result:
                inserted_ids.append(essay_id)
        return inserted_ids
    except Exception as e:
        logging.error(f"Error inserting essay questions: {e}")
        return {"status": "error", "message": str(e)}


async def insert_multichoice_questions(package_id, questions, answers, explanations=None):
    """
    Insert the series of multichoice questions
    """
    if len(questions) != len(answers):
        return {"status": "error", "message": "Questions and answers must have the same length"}

    if explanations and len(explanations) != len(questions):
        return {"status": "error", "message": "Explanations must have the same length as questions if provided"}

    query = f"""
    INSERT INTO {POSTGRES_SCHEMA}.quiz_multichoice (multichoice_id, package, question, answer, explanation, is_deleted)
    VALUES ($1, $2, $3, $4, $5, $6) RETURNING multichoice_id
    """
    try:
        inserted_ids = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            explanation = explanations[i] if explanations else ""
            multichoice_id = str(uuid.uuid4())  # Generate UUID locally
            result = await db_util.post_data(
                query,
                multichoice_id,
                package_id,
                question,
                answer,
                explanation,
                False
            )
            if result:
                inserted_ids.append(multichoice_id)
        return inserted_ids
    except Exception as e:
        logging.error(f"Error inserting multichoice questions: {e}")
        return {"status": "error", "message": str(e)}


async def insert_multichoice_questions():
    return None


async def fetch_evaluation(evaluation_id):
    """
    Fetch evaluation details by evaluation_id.
    """
    query = f"SELECT * FROM {POSTGRES_SCHEMA}.evaluation WHERE eval_id = $1"
    try:
        data = await db_util.fetch_data(query, evaluation_id)
        return dict(data[0]) if data else None

    except Exception as e:
        logging.error(f"Error fetching evaluation: {e}")
        return None


async def fetch_question(question_id):
    """
    Fetch queation details by question_id.
    """
    query = f"SELECT * FROM {POSTGRES_SCHEMA}.quiz_essay WHERE essay_id = $1"

    try:
        data = await db_util.fetch_data(query, question_id)
        return dict(data[0]) if data else None

    except Exception as e:
        logging.info(f"Error fetching evaluation: {e}")
        return None


async def update_evaluation(evaluation_id, score):
    """
    Update score by evaluation id.
    """
    query = f"UPDATE {POSTGRES_SCHEMA}.evaluation SET score = $1 WHERE eval_id = $2 RETURNING eval_id"
    try:
        result = await db_util.post_data(query, score, evaluation_id)
        return {"status": "success", "message": "Evaluation updated successfully"} if result else {"status": "error", "message": "Evaluation not found"}
    except Exception as e:
        print(f"Error updating evaluation: {e}")
        return {"status": "error", "message": str(e)}


async def update_quiz_activities(quiz_id, final_score):
    """
    Update final score by quiz id.
    """
    query = f"UPDATE {POSTGRES_SCHEMA}.quiz_activities SET final_score = $1 WHERE quiz_id = $2 RETURNING quiz_id"
    try:
        result = await db_util.post_data(query, final_score, quiz_id)
        return {"status": "success", "message": "Quiz activity updated successfully"} if result else {"status": "error", "message": "Quiz activity not found"}
    except Exception as e:
        print(f"Error updating quiz activity: {e}")
        return {"status": "error", "message": str(e)}

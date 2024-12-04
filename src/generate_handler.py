import asyncio
import logging
import os
from db_handler import fetch_package, insert_essay_questions, insert_multichoice_questions
from inference_jobs.generate import generate_quiz_question


async def generate_request_handler(data):
    """
    Handling inferencing process

    1. Fetching package -> get prompt, get context (PDF)
    2. Do the quiz generating process 
    3. Update questions
    4. Delete context (PDF)
    """
    try:
        # Step 1: Fetching package
        package_id = data.get("packageId")
        package = await fetch_package(package_id)

        if not package:
            return {"status": "error", "message": f"Package {package_id} not found."}

        # Check for valid PDF context
        if "pdf_path" not in package:
            return {"status": "error", "message": "No valid PDF context found in package."}

        # Step 2: Generating quiz
        questions, options, answers = await generate_quiz_question(package)

        # Step 3: Update questions
        if package.get("type") == 0:  # Multi-choice questions
            question_ids = await insert_multichoice_questions(package_id, questions, answers, options)
            question_data = [
                {
                    "question_id": str(qid),
                    "question": q,
                    "options": opt,  
                    "answer": ans
                }
                for qid, q, opt, ans in zip(question_ids, questions, options, answers)
            ]

        elif package.get("type") == 1:  # Essay questions
            question_ids = await insert_essay_questions(package_id, questions, answers)
            question_data = [
                {
                    "question_id": str(qid),
                    "question": q,
                    "answer": ans
                }
                for qid, q, ans in zip(question_ids, questions, answers)
            ]

        else:
            return {"status": "error", "message": "Invalid package type."}

        # Step 4: Delete context
        pdf_context_path = package.get("pdf_path")
        if os.path.exists(pdf_context_path):
            os.remove(pdf_context_path)

 

        return {"package_id": package_id, "question_data": question_data}

    except Exception as e:
        logging.error(f"Error in generate_request_handler: {e}")
        return {"status": "error", "message": f"An error occurred: {e}"}

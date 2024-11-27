import asyncio
import logging
from db_handler import fetch_evaluation, fetch_question, update_evaluation, update_quiz_activities
from inference_jobs.evaluate import calculate_essay_score


async def evaluate_request_handler(data):
    """
    Handling inferencing process

    This function includes the loop for n evaluations from the eval list, performing:
    1. Fetching evaluation -> get user_answer, question_id
    2. Fetching question -> get context by question_id
    3. Do the evaluation process
    4. Retrieve score
    5. Update evaluation -> set score
    6. Update quiz activities
    """

    evaluation_list = data.get("evalIdList")
    if not evaluation_list:
        return {"status": "error", "message": "evaluation_id is missing in the request"}

    user_answers = []
    contexts = []
    eval_ids = []
    question_ids = []

    # Gather user answers and contexts asynchronously
    async def gather_data():
        async def fetch_data(eval_id):
            try:
                # Fetch evaluation
                eval_data = await fetch_evaluation(eval_id)
                user_answer = eval_data.get("user_answer")
                question_id = eval_data.get("question_id")

                if not user_answer or not question_id:
                    raise ValueError(
                        f"Missing user_answer or question_id for eval_id {eval_id}")

                # Fetch question
                question_data = await fetch_question(question_id)
                context = question_data.get("context")

                if not context:
                    raise ValueError(
                        f"Missing context for question_id {question_id}")

                # Append data to lists
                user_answers.append(user_answer)
                contexts.append(context)
                eval_ids.append(eval_id)
                question_ids.append(question_id)

            except Exception as e:
                logging.error(
                    f"Error fetching data for eval_id {eval_id}: {e}")

        # Concurrently fetch data
        tasks = [fetch_data(eval_id) for eval_id in evaluation_list]
        await asyncio.gather(*tasks)

    # Run the async process
    try:
        # Step 1 and 2: Gather all user answers and contexts
        await gather_data()

        if not user_answers or not contexts:
            return {"status": "error", "message": "Failed to gather user answers or contexts"}

        # Step 3: Perform evaluation
        score_list, final_score = await calculate_essay_score(user_answers, contexts)

        # Step 4: Update evaluation
        async def update_data():
            tasks = [
                update_evaluation(eval_id, score)
                for eval_id, score in zip(eval_ids, score_list)
            ]
            await asyncio.gather(*tasks)

        await update_data()

        # Step 5: Update quiz activities
        quiz_id = data.get("quizId")
        if quiz_id:
            await update_quiz_activities(quiz_id, final_score)

        # Step 6: Formatting the output
        eval_data = [{"eval_id": str(eval_id), "question_id": str(question_id), "score": score}
                     for eval_id, question_id, score in zip(eval_ids, question_ids, score_list)]
        avg_final_score = round((final_score / len(data.get("evalIdList"))), 2)

        return {"quiz_id": data.get("quizId"), "eval_data": eval_data, "final_score": avg_final_score}

    except Exception as e:
        logging.error(f"Error during evaluation process: {e}")
        return {"status": "error", "message": str(e)}

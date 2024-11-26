import asyncio
import logging
from db_handler import fetch_evaluation, fetch_question, update_evaluation


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

    # Process evaluations asynchronously
    async def process_evaluations():
        async def process_eval(eval_id):
            try:
                # Step 1: Fetch evaluation
                eval_data = await fetch_evaluation(eval_id)
                logging.info(f"Evaluation fetched for {eval_id}: {eval_data}")
                return {"evaluation_id": eval_id, "status": "success", "data": eval_data}

            except Exception as e:
                logging.error(f"Error fetching evaluation for {eval_id}: {e}")
                return {"evaluation_id": eval_id, "status": "error", "message": str(e)}

        # Use asyncio.gather for concurrent execution
        tasks = [process_eval(eval_id) for eval_id in evaluation_list]
        return await asyncio.gather(*tasks)

    # Run the async process
    try:
        # Directly await instead of using run_until_complete
        result = await process_evaluations()
        return {"status": "success", "results": "success"}

    except Exception as e:
        logging.error(f"Error during evaluation process: {e}")
        return {"status": "error", "message": str(e)}

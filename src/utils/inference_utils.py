import json
import logging
import threading
import time
import os
from utils.gpu_utils import pick_gpus_by_memory_usage
from generate_handler import generate_request_handler
from evaluate_handler import evaluate_request_handler

# TODO: Terminate whole inference process if theres no new message
# TODO: GPU allocation


class InferenceProcessManager:
    def __init__(self, gpu_count=2):
        self.last_used = None
        self.lock = threading.Lock()
        self.shutdown_timer = None
        self.inactivity_timeout = 10 * 60  # 10 minutes in seconds

    async def start(self, data):
        """Start the inference process and directly call inference handler"""
        with self.lock:
            logging.info("Starting inference process...")

            # Select GPUs with the least memory usage
            available_gpus = pick_gpus_by_memory_usage(
                count=2)  # Adjust count as needed
            if available_gpus:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(available_gpus)
                logging.info(f"Assigned GPUs: {available_gpus}")
            else:
                logging.warning("No available GPUs. Proceeding with CPU.")

            # Call the inference functionality
            result = await self._call_generate(data)

            return result

    async def _call_generate(self, data):
        """Directly call inference function from handler.py."""
        try:
            req_type = data.get("reqType")
            if req_type is None:
                logging.warning(
                    "No request provided; skipping inference call.")
                return {"status": "error", "message": "No request provided"}

            # Invoking generate quiz inference
            if req_type == 2:
                logging.info(f"Invoking essay quiz evaluation inference type")
                result = await evaluate_request_handler(data)

            # Invoking evaluating essay inference
            else:
                logging.info(f"Invoking quiz generating inference type")
                result = await generate_request_handler(data)

            logging.info(f"Inference result: {result}")

            return result
        except Exception as e:
            logging.error(f"Error calling inference handler: {e}")
            return {"status": "error", "message": str(e)}

import json
import logging
import threading
import time
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

            # Call the inference functionality
            result = await self._call_generate(data)

            logging.info("Finish 1 inference process...")

            # # Update the last used time for shutdown timer
            # self.last_used = time.time()

            # # Reset the shutdown timer
            # self.reset_shutdown_timer()

            return result

    # def stop(self):
    #     """Stop any activity and reset the manager state."""
    #     with self.lock:
    #         logging.info("Stopping inference process manager...")
    #         self.last_used = None
    #         if self.shutdown_timer:
    #             self.shutdown_timer.cancel()

    # def reset_shutdown_timer(self):
    #     """Reset the shutdown timer to stop the manager after inactivity."""
    #     with self.lock:
    #         self.last_used = time.time()

    #         # Cancel any existing shutdown timer
    #         if self.shutdown_timer:
    #             self.shutdown_timer.cancel()

    #         # Set a new timer
    #         self.shutdown_timer = threading.Timer(
    #             self.inactivity_timeout, self._check_inactivity
    #         )
    #         self.shutdown_timer.start()

    # def _check_inactivity(self):
    #     """Check if the manager has been inactive and stop it if needed."""
    #     with self.lock:
    #         if time.time() - self.last_used >= self.inactivity_timeout:
    #             logging.info(
    #                 "No activity for 10 minutes. Stopping inference process.")
    #             self.stop()

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

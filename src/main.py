import asyncio
import json
import logging
import os
from aio_pika import connect_robust, Message, ExchangeType
from aio_pika.abc import AbstractIncomingMessage
from dotenv import load_dotenv
from utils.inference_utils import InferenceProcessManager
import torch
import subprocess


# Load environment variables
load_dotenv()

# Configure logging
log_format = "%(asctime)s [%(levelname)s]: (%(name)s) %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: (%(name)s) %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logging.getLogger('torch').setLevel(logging.INFO)


# Check if CUDA is available
if torch.cuda.is_available():
    logging.info("CUDA is available. Activating GPU.")
    torch.set_default_device("cuda")
else:
    logging.info("CUDA is not available. Falling back to CPU.")
    torch.set_default_device("cpu")


# RabbitMQ configuration from .env
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE")
RABBITMQ_DIRECT_MESSAGE = os.getenv("RABBITMQ_DIRECT_EXCHANGE")

# Initialize the inference process manager
inference_manager = InferenceProcessManager()


async def log_gpu_usage(selected_gpus):
    """Log usage statistics for the chosen GPUs periodically."""
    while True:
        try:
            # Use NVML for more precise GPU monitoring
            import pynvml
            pynvml.nvmlInit()

            # Log details for each selected GPU
            for gpu_id in selected_gpus:
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                logging.info(
                    f"GPU {gpu_id}: "
                    f"Memory Used: {memory.used /
                                    1e6:.2f} MB / {memory.total / 1e6:.2f} MB, "
                    f"Utilization: {utilization.gpu}%"
                )
            pynvml.nvmlShutdown()

        except Exception as e:
            logging.error(f"Error while logging GPU usage: {e}")

        # Wait for a few seconds before the next log
        await asyncio.sleep(30)


async def task(message: AbstractIncomingMessage, channel):
    """Task to handle incoming RabbitMQ messages."""
    try:
        # Decode and log the received message
        data = json.loads(message.body.decode())
        logging.info(f"Received message: {data}")

        # Process the request
        response_data = await inference_manager.start(data)

        # Launch GPU monitoring for the selected GPUs
        # asyncio.create_task(log_gpu_usage(selected_gpus))

        # Check for errors in the inference result
        if response_data.get("status") == "error":
            logging.error(f"Inference failed: {response_data['message']}")
            raise RuntimeError("Inference process failed")

        logging.info(f"Processed successfully: {response_data}")

        # Send response to the reply-to queue
        if message.reply_to:
            await channel.default_exchange.publish(
                Message(
                    body=json.dumps(response_data).encode(),
                    content_type="application/json",
                    correlation_id=message.correlation_id,  # Ensure correlation_id is sent back
                ),
                routing_key=message.reply_to,  # Use the reply_to queue provided by the client
            )
            logging.info("Response sent successfully.")

        # Explicitly acknowledge the message
        await message.ack()
        logging.info("Message acknowledged.")

    except Exception as e:
        logging.error(f"Error processing message: {e}")

        # Prepare error response
        error_response = {
            "status": "error",
            "message": str(e),
            "details": "An error occurred while processing the message."
        }

        # Send error response to the reply-to queue if available
        if message.reply_to:
            await channel.default_exchange.publish(
                Message(
                    body=json.dumps(error_response).encode(),
                    content_type="application/json",
                    correlation_id=message.correlation_id,  # Ensure correlation_id is sent back
                ),
                routing_key=message.reply_to,  # Use the reply_to queue provided by the client
            )
            logging.info("Error response sent successfully.")

        # Negatively acknowledge the message; choose whether to requeue
        await message.nack(requeue=False)
        logging.error(
            "Message negatively acknowledged and will not be requeued.")


async def main():
    """Main function to establish RabbitMQ connection and start consuming messages."""
    connection = await connect_robust(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        login=RABBITMQ_USER,
        password=RABBITMQ_PASSWORD,
    )

    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    # Declare exchange
    exchange = await channel.declare_exchange(RABBITMQ_DIRECT_MESSAGE, ExchangeType.DIRECT, durable=True)

    # Declare queue
    queue = await channel.declare_queue(RABBITMQ_QUEUE, durable=False)

    # Bind queue to exchange
    await queue.bind(exchange, routing_key="rpc")

    # Start consuming and pass the channel to the task function

    queue.consume(lambda message: task(message, channel))
    # log_gpu_usage()  # Add the GPU monitoring task

    logging.info("Waiting for messages...")

    try:
        await asyncio.Future()
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())

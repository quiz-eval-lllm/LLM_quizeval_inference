import pynvml
import logging


def pick_gpus_by_memory_usage(count=2):
    """
    Select GPUs with the least memory usage.

    Args:
        count (int): The number of GPUs to select.

    Returns:
        List[str]: A list of GPU indices as strings, sorted by memory usage.
    """
    try:
        # Initialize NVML
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count == 0:
            logging.warning("No GPUs detected on the system.")
            return []

        # Adjust count to the available number of GPUs
        count = min(count, device_count)

        memory_info = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_usage = memory.used / memory.total  # Percentage of memory used
            memory_info.append((i, memory_usage, memory.total, memory.used))

        # Sort GPUs by memory usage (ascending)
        memory_info.sort(key=lambda x: x[1])

        selected_gpus = [str(gpu[0]) for gpu in memory_info[:count]]
        return selected_gpus

    except pynvml.NVMLError as e:
        logging.error(f"NVML error: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return []
    finally:
        # Ensure NVML is properly shut down
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            logging.warning(f"Error shutting down NVML: {str(e)}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    # Number of GPUs to select
    num_gpus = 2  # Change this value to select a different number of GPUs

    # Run the function and print the selected GPUs
    selected_gpus = pick_gpus_by_memory_usage(count=num_gpus)
    if selected_gpus:
        print(f"Selected GPUs: {', '.join(selected_gpus)}")
    else:
        print("No GPUs selected or an error occurred.")

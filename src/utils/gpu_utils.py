import pynvml


def pick_gpus_by_memory_usage(count=2):
    """
    Pick GPUs with the least memory usage.

    Args:
        count (int): Number of GPUs to select.

    Returns:
        List[str]: List of GPU indices with the least memory usage.
    """
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    memory_info = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # Calculate percentage of memory used
        memory_usage = memory.used / memory.total
        memory_info.append((i, memory_usage))

    # Sort GPUs by memory usage (ascending)
    memory_info.sort(key=lambda x: x[1])
    selected_gpus = [str(gpu[0]) for gpu in memory_info[:count]]

    pynvml.nvmlShutdown()
    return selected_gpus

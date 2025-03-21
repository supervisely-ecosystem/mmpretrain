import supervisely as sly


def get_gpu_devices():
    try:
        from torch import cuda
    except ImportError as ie:
        sly.logger.warning(
            "Unable to import Torch. Please, run 'pip install torch' to resolve the issue.",
            extra={"error message": str(ie)},
        )
        return [{"label": "cuda:0", "value": "0", "right_text": None}]

    try:
        cuda.init()
        if not cuda.is_available():
            raise RuntimeError("CUDA is not available")
    except Exception as e:
        sly.logger.warning(f"Failed to initialize CUDA: {e}")
        return [{"label": "cuda:0", "value": "0", "right_text": None}]
    try:
        devices = []
        for idx in range(cuda.device_count()):
            current_device = f"cuda:{idx}"
            full_device_name = f"{cuda.get_device_name(idx)} ({current_device})"
            free_m, total_m = cuda.mem_get_info(current_device)

            convert_to_gb = lambda number: round(number / 1024**3, 1)
            right = f"{convert_to_gb(total_m - free_m)} GB / {convert_to_gb(total_m)} GB"

            devices.append({"label": full_device_name, "value": str(idx), "right_text": right})
        return devices
    except Exception as e:
        sly.logger.warning(repr(e))
        return [{"label": "cuda:0", "value": "0", "right_text": None}]

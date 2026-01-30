"""GPU management utilities"""

import os
import subprocess
from typing import List, Optional
from .logging_utils import get_logger

logger = get_logger(__name__)

class GPUManager:
    """Manage GPU allocation and usage"""
    
    def __init__(self):
        self.available_gpus = self.detect_gpus()
    
    def detect_gpus(self) -> List[int]:
        """Detect available GPUs"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, check=True)
            gpu_ids = [int(gpu_id.strip()) for gpu_id in result.stdout.strip().split('\n') if gpu_id.strip()]
            # logger.info(f"Detected GPUs: {gpu_ids}")
            return gpu_ids
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not detect GPUs, defaulting to GPU 0")
            return [0]
    
    # def set_gpu(self, gpu_id: int):
    #     """Set CUDA_VISIBLE_DEVICES for a specific GPU"""
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    #     logger.info(f"Set CUDA_VISIBLE_DEVICES to {gpu_id}")
    
    # def allocate_gpus(self, num_jobs: int, max_gpus: int = 5) -> List[int]:
    #     """Allocate GPUs for batch processing"""
    #     user_gpu_list=os.environ['CUDA_VISIBLE_DEVICES']
    #     print("USER GPU LIST:", user_gpu_list)
    #     if user_gpu_list:
    #         # invalid_gpus = [gpu for gpu in user_gpu_list if gpu not in self.available_gpus]
    #         # if invalid_gpus:
    #         #     logger.warning(f"GPUs {invalid_gpus} are not available")
    #         gpu_list = [gpu for gpu in user_gpu_list if gpu in self.available_gpus]
    #     else:
    #         gpu_list = self.available_gpus[:min(num_jobs, max_gpus, len(self.available_gpus))]
        
    #     logger.info(f"Allocated GPUs: {gpu_list} for {num_jobs} jobs")
    #     return gpu_list
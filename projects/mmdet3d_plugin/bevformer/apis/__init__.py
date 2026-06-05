from .train import custom_train_model
from .mmdet_train import custom_train_detector
from .test import single_gpu_test
# from .test import custom_multi_gpu_test

__all__ = [
    'single_gpu_test'
]
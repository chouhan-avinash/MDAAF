# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_segmentor, init_segmentor, show_result_pyplot
from .test import multi_gpu_test, multi_gpu_test1, single_gpu_test,single_gpu_test2_isave,single_gpu_test2_val,single_gpu_test_loveda,single_gpu_test_is
#from .test2_isprs_save import single_gpu_test2_isave
from .train import (get_root_logger, init_random_seed, set_random_seed,
                    train_segmentor)

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_segmentor', 'init_segmentor',
    'inference_segmentor', 'multi_gpu_test', 'multi_gpu_test1','single_gpu_test','single_gpu_test_loveda','single_gpu_test2_isave','single_gpu_test2_val',

    'show_result_pyplot', 'init_random_seed','single_gpu_test_is'
]

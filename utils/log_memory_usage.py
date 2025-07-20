import torch
import pytorch_lightning as pl
import torch
import inspect
import textwrap
# import pynvml
from scalene import scalene_profiler
from pytorch_memlab import LineProfiler
from pytorch_memlab import MemReporter

def log_gpu_memory_usage(trainer: pl.LightningModule):
        # log memory usage
        trainer.log("device_memory_usage/%s_max"%str(trainer.device), torch.cuda.max_memory_allocated()/(1024*1024))
        trainer.log("device_memory_usage/%s"%str(trainer.device), torch.cuda.memory_allocated()/(1024*1024))
        # torch.cuda.memory_cached()
        # pynvml.nvmlInit()
        # device = trainer.device
        # if device.type == "cuda":
        #     cuda_id = device.index
        #     handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_id)
        #     info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        #     total_memory = info.total
        #     used_memory = info.used
        #     free_memory = info.free
        #     trainer.log("device_memory_usage/%s_pynvml"%str(trainer.device), used_memory)
        
        

def profile_cuda_memory(func):
        def wrapper(*args, **kwargs):
                
                profiler = LineProfiler()
                profiler.add_function(func)
                profiler.enable()
                func(*args, **kwargs)
                with open("memory_usage.log", "a+") as f:
                        profiler.print_stats(stream=f)
                profiler.print_stats()
                profiler.disable()
        return wrapper


if __name__ == "__main__":
        @profile_cuda_memory
        def my_test():
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.randn(1000, 1000, device='cuda')
                z = x + y
                w = z * 2
        # scalene_profiler.start()

        # your code
        # my_test()

        # Turn profiling off
        # scalene_profiler.stop()
        
        # profiler = LineProfiler()
        # profiler.add_function(my_test)
        # profiler.enable()
        my_test()
        # profiler.print_stats()
        

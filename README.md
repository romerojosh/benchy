# benchy
Benchmarking Dataloader Wrapper for DL

## What is it?
`benchy` is a simple tool that can capture and report throughput metrics for DL
training workloads through wrapping your dataloading iterator. When used, the tool will
run, measure, and report the throughput of the following in samples per second:
1. `IO`: your dataloader running in isolation
2. `SYNTHETIC`: your training workload when provided synthetic (or cached) data samples
3. `FULL`: your training worload when provided real data samples

Comparing these thoughputs can help highlight what is bottlenecking your workload and help
focus optimization efforts.

This tool is being used for the Deep Learning at Scale tutorial at SC21 ([link](https://github.com/NERSC/sc21-dl-tutorial)). However,
it could be useful in other workloads and is available here.

## How to use?
The tool currently supports PyTorch Dataloader iterators and other similar Python iterators (e.g.
iterators from the NVIDIA DALI library).
For PyTorch dataloaders, `benchy.torch.BenchmarkDataLoader` can be used as a stand-in replacement as follows:
* Using `torch.utils.data.DataLoader`:
```
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size)
```
* Using `benchy.torch.BenchmarkDataLoader`:
```
from benchy.torch import BenchmarkDataLoader
train_loader = BenchmarkDataLoader(dataset, batch_size)
```

For other Python iterators, you can use `benchy.torch.BenchmarkGenericIteratorWrapper` as follows:
```
train_loader = CustomDataLoader(dataset, batch_size)
train_loader = benchy.torch.BenchmarkGenericIteratorWrapper(train_loader, batch_size)
```

With this in place, `benchy` will override the dataloader behavior to generate throughput numbers for the `IO`,
`SYNTHETIC`, and `FULL` scenarios. At the end of a successful run with this tool, a summary will be printed to
the terminal reporting the measured throughputs:
```
BENCHY::SUMMARY::IO average throughput: 8.808 +/- 0.132
BENCHY::SUMMARY:: SYNTHETIC average throughput: 19.253 +/- 0.035
BENCHY::SUMMARY::FULL average throughput: 8.465 +/- 0.211
```
Additionally, a JSON file will be output containing measured throughput values for postprocessing/plotting.

See `sample_benchy_conf.yaml` for available configuration options (e.g. number of trials to run, report frequency, etc.) and
`benchy/__init__.py:_get_default_config` for defaults. To override defaults, set environment variable
`BENCHY_CONFIG_FILE=<your modified configfile>` when running.

Note that each trial in `benchy` acts like a full training epoch in your script. For correct behavior, set your training
options to perform enough epochs to cover the number of trial and warmup trials requested in your configuration.

## Profiling
Besides throughput measurements, `benchy` also has useful features for NVIDIA Nsight Systems command-line profiling (`nsys profile`):
1. Adds NVTX annotations to label training iterations, data loading time, and the duration of the different measured trials.
2. Controls if profiling is started on a single or all GPUs when used with the `--capture-range cudaProfilerApi` flag (see `profiler_mode` configuration option).
This can be useful when running multi-GPU and you want to limit the profiling output.

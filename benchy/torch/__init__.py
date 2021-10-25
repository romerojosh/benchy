# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from benchy import _initialize, _finalize, _framework

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

import time

def _barrier(dist_barrier=False):
  torch.cuda.synchronize()
  if dist_barrier and dist.is_available() and dist.is_initialized():
    dist.barrier()

def _init_self(self):
    self.rank = 0
    self.comm_size = 1
    if dist.is_available() and dist.is_initialized():
      self.rank = dist.get_rank()
      self.comm_size = dist.get_world_size()

    self.benchy_config = _initialize(self.rank, _framework.PYTORCH)

    self.synth_count = 0
    self.full_count = 0

    self.results = {}
    self.results["nranks"] = self.comm_size

class _BenchmarkIterator:
  def __init__(self, it, batch_size, report_freq, nbatches, label,
               results, cached, dist_barrier):
    self.it = it
    self.count = 0
    self.sample_count = 0
    self.t0 = None
    self.rates = []
    self.report_freq = report_freq
    self.batch_size = batch_size
    self.cached = cached
    self.dist_barrier = dist_barrier
    self.cached_output = None
    try:
      self.nbatches = len(it) if not nbatches else min(len(it), nbatches)
    except:
      self.nbatches = nbatches

    self.label = label
    self.skip_results = "WARMUP" in label

    self.results = results
    if self.label.split('_')[0] not in self.results.keys() and not self.skip_results:
      self.results[self.label.split('_')[0]] = {"rates" : [], "avg_throughput" : []}

    self.rank = 0
    self.comm_size = 1

    if dist.is_available() and dist.is_initialized():
      self.rank = dist.get_rank()
      self.comm_size = dist.get_world_size()

  def __iter__(self):
    return self

  def __next__(self):
    if self.cached and not self.cached_output:
      self.cached_output = next(self.it)
      return self.cached_output

    if not self.t0:
      _barrier(self.dist_barrier)
      self.t0 = time.time()
      torch.cuda.nvtx.range_push(f"BENCHY::BenchmarkIterator::{self.label}")
      torch.cuda.nvtx.range_push(f"BENCHY::BenchmarkIterator::{self.label}::STEP_{self.count}")
    else:
      _barrier(self.dist_barrier)
      torch.cuda.nvtx.range_pop() # STEP
      torch.cuda.nvtx.range_push(f"BENCHY::BenchmarkIterator::{self.label}::STEP_{self.count}")
      t1 = time.time()

      rate = (self.batch_size * self.comm_size) / (t1 - self.t0)
      self.rates.append(rate)
      self.t0 = t1
      if self.sample_count == self.report_freq:
        if self.rank == 0:
          print(f"BENCHY::{self.label}::Throughput: {rate}")
        self.sample_count = 0

      if self.count == self.nbatches:
        if self.rank == 0:
          average_rate = sum(self.rates)/len(self.rates)
          print(f"BENCHY::{self.label}::Avg Throughput: {average_rate}")
          torch.cuda.nvtx.range_pop() # STEP
          torch.cuda.nvtx.range_pop() # MAIN
          if not self.skip_results:
            self.results[self.label.split('_')[0]]["rates"].append(self.rates)
            self.results[self.label.split('_')[0]]["avg_throughput"].append(average_rate)
        raise StopIteration

    if self.cached:
      output = self.cached_output
    else:
      try:
        torch.cuda.nvtx.range_push(f"BENCHY::BenchmarkIterator::{self.label}::DATA_{self.count}")
        output = next(self.it)
        torch.cuda.nvtx.range_pop()
      except StopIteration:
        print("stopped!")
        torch.cuda.nvtx.range_pop() # STEP
        torch.cuda.nvtx.range_pop() # MAIN
        raise StopIteration

    self.count += 1
    self.sample_count += 1

    return output

def _run_io_bench(self, it_func):
  if self.benchy_config["IO"]["run_benchmark"]:
    nt = self.benchy_config["IO"]["ntrials"]
    nw = self.benchy_config["IO"]["nwarmup"]
    for n in range(nt + nw):
      it = it_func()
      label = f"IO_{n - nw}" if n >= nw else f"IO_WARMUP_{n}"
      it = _BenchmarkIterator(it, self.batch_size, report_freq=self.benchy_config["global"]["report_freq"],
                             nbatches=self.benchy_config["IO"]["nbatches"], label=label,
                             results=self.results, cached=False,
                             dist_barrier=self.benchy_config["global"]["use_distributed_barrier"])
      for _ in it:
        pass

def _run_synth_full_bench(self, it):
    if (self.benchy_config["synthetic"]["run_benchmark"] and
       self.synth_count < self.benchy_config["synthetic"]["ntrials"] +  self.benchy_config["synthetic"]["nwarmup"]):
      nt = self.benchy_config["synthetic"]["ntrials"]
      nw = self.benchy_config["synthetic"]["nwarmup"]
      label = f"SYNTHETIC_{self.synth_count - nw}" if self.synth_count >= nw else f"SYNTHETIC_WARMUP_{self.synth_count}"
      it = _BenchmarkIterator(it, self.batch_size, report_freq=self.benchy_config["global"]["report_freq"],
                             nbatches=self.benchy_config["synthetic"]["nbatches"], label=label,
                             results=self.results, cached=True,
                             dist_barrier=self.benchy_config["global"]["use_distributed_barrier"])
      self.synth_count += 1
      return it

    if (self.benchy_config["full"]["run_benchmark"] and
       self.full_count < self.benchy_config["full"]["ntrials"] +  self.benchy_config["full"]["nwarmup"]):
      nt = self.benchy_config["full"]["ntrials"]
      nw = self.benchy_config["full"]["nwarmup"]
      label = f"FULL_{self.full_count - nw}" if self.full_count >= nw else f"FULL_WARMUP_{self.full_count}"
      it = _BenchmarkIterator(it, self.batch_size, report_freq=self.benchy_config["global"]["report_freq"],
                             nbatches=self.benchy_config["full"]["nbatches"], label=label,
                             results=self.results, cached=False,
                             dist_barrier=self.benchy_config["global"]["use_distributed_barrier"])
      self.full_count += 1
      return it
    return None

class BenchmarkDataLoader(DataLoader):
  def __init__(self, *args, **kwargs):
    super(BenchmarkDataLoader, self).__init__(*args, **kwargs)

    _init_self(self)

    it_func = super(BenchmarkDataLoader, self).__iter__
    _run_io_bench(self, it_func)

  def __iter__(self):
    it = super(BenchmarkDataLoader, self).__iter__()
    it_out = _run_synth_full_bench(self, it)
    if it_out:
      return it_out

    _finalize(self)

    return it

class BenchmarkGenericIteratorWrapper(object):
  def __init__(self, iterator, batch_size):
    self.iterator = iterator
    self.batch_size = batch_size

    _init_self(self)

    it_func = self.iterator.__iter__
    _run_io_bench(self, it_func)

  def __iter__(self):
    it = self.iterator.__iter__()
    it_out = _run_synth_full_bench(self, it)
    if it_out:
      return it_out

    _finalize(self)

    return it

  def __len__(self):
    return len(self.iterator)

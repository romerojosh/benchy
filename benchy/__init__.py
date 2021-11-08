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

from datetime import datetime
import enum
import json
import os
import yaml
import sys

import numpy as np

class _framework(enum.Enum):
    PYTORCH = 1

def _get_default_config():
  config = {'global': {'report_freq': 10,
                        'exit_after_tests': True,
                        'profiler_mode': 'single',
                        'output_filename': 'benchy_result.json',
                        'output_dir': os.getcwd(),
                        'use_distributed_barrier': False,
                       },
            'IO': {'run_benchmark': True,
                   'nbatches': 50,
                   'ntrials': 3,
                   'nwarmup': 1,
                  },
            'synthetic': {'run_benchmark': True,
                   'nbatches': 50,
                   'ntrials': 3,
                   'nwarmup': 1,
                  },
            'full': {'run_benchmark': True,
                   'nbatches': 50,
                   'ntrials': 3,
                   'nwarmup': 1,
                  },
          }
  return config

def _print_results_summary(results):
  if 'IO' in results.keys():
    io_avg = np.mean(results['IO']['trial_throughput'])
    io_std = np.std(results['IO']['trial_throughput'])
    print("BENCHY::SUMMARY::IO average trial throughput: {0:.3f} +/- {1:.3f}".format(io_avg, io_std))
  if 'SYNTHETIC' in results.keys():
    synth_avg = np.mean(results['SYNTHETIC']['trial_throughput'])
    synth_std = np.std(results['SYNTHETIC']['trial_throughput'])
    print("BENCHY::SUMMARY:: SYNTHETIC average trial throughput: {0:.3f} +/- {1:.3f}".format(synth_avg, synth_std))
  if 'FULL' in results.keys():
    full_avg = np.mean(results['FULL']['trial_throughput'])
    full_std = np.std(results['FULL']['trial_throughput'])
    print("BENCHY::SUMMARY::FULL average trial throughput: {0:.3f} +/- {1:.3f}".format(full_avg, full_std))

def _initialize(rank, framework):
    benchy_config = _get_default_config()
    benchy_config_file = os.environ.get('BENCHY_CONFIG_FILE')
    if benchy_config_file:
      if rank == 0:
        print(f"BENCHY::INFO::Message: Using benchy configuration file {benchy_config_file}.")
      with open(benchy_config_file, "r") as f:
        user_benchy_config = yaml.safe_load(f)
      benchy_config.update(user_benchy_config)
    else:
      if rank == 0:
        print(f"BENCHY::INFO::Message: Using default benchy configuration.")

    # Start nsys
    prof_mode = benchy_config["global"]["profiler_mode"]
    if prof_mode == "single":
      if rank == 0:
        print(f"BENCHY::INFO::Message: Profiling mode single. Starting profiler on rank 0 only.")
        if framework == _framework.PYTORCH:
          import torch.cuda.profiler as profiler
          profiler.start()
    elif prof_mode == "all":
      if rank == 0:
        print(f"BENCHY::INFO::Message: Profiling mode all. Starting profiler on all ranks.")
      if framework == _framework.PYTORCH:
        import torch.cuda.profiler as profiler
        profiler.start()

    return benchy_config

def _finalize(self):
    if self.rank == 0:
      _print_results_summary(self.results)

      output_dir = self.benchy_config["global"]["output_dir"]
      os.makedirs(output_dir, exist_ok=True)

      json_file = self.benchy_config["global"]["output_filename"]
      json_file = os.path.join(output_dir, json_file)
      print(f"BENCHY::INFO::Message: Writing JSON output to {json_file}.")
      with open(json_file, 'w') as f:
        json.dump(self.results, f)

    if self.benchy_config["global"]["exit_after_tests"]:
      if self.rank == 0:
        print(f"BENCHY::INFO::Message: Tests completed.... exiting.")
      sys.exit()

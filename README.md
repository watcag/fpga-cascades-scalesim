# Scaling the Cascades -- Simulation using SCALE-Sim

This documentation provides insight into the simulation infrastructure used for the paper "Scaling the Cascades" 
using the open source CNN accelerator simulator, SCALE-Sim.

>    *Ananda Samajdar, Tushar Garg, Tushar Krishna, and Nachiket Kapre*

>    [**"Scaling the Cascades: Interconnect-aware FPGA implementation of Machine Learning problems"**](https://nachiket.github.io/publications/stc_fpl-2019.pdf),

>    International Conference on Field-Programmable Logic and Applications, Sep 2019 

## Modelling DSP cascades in Xilinx Ultrascale using SCALE-Sim

The detailed model of DNN acceleration infrastructure using the DSP cascades can be found in the [paper](https://nachiket.github.io/publications/stc_fpl-2019.pdf). 
Functionally however, the architecture of the accelerator can be approximated as a 9xN systolic array, where N is the 
number of processors. The maximum number of processors that could be supported in the VU37P FPGA is 960. 

In this work our each of these processors can be configured as either Matrix-Multiply(MM) or Convolution (Conv) processors. The entire 
DSP cascade can therefore work either completely as a matrix-multiply or convolution unit, or as a proportional mix of 
the two configurations. 
Modelling the Convolution or Matrix-Mulitply unit onto a systolic array is equivalent to changing the dataflow. 
Convolution processors are equivalent to a systolic array using a Weight Stationary (WS) dataflow, while Matrix-Multiply
is equivalent to Input Stationary (IS) dataflow.
Moreover, Conv processors can be configured to operate in SIMD mode, which effectively double the number of columns in 
the systolic array equivalent.

## Cascades as scaled out DNN accelerator
It is natural that a very shallow yet wide monolithic systolic array would lead to high under-utilization and therefore
degraded performance on any modern deep neural network. To combat this issue the FPGA is configured to be functionally
equivalent to a scaled out cluster of processors with many equivalent systolic arrays working in parallel. 

The papers details how data is transferred to keep all the units fed at runtime, and the mechanism to collect the outputs. 
For simulations to generate runtime, we split up the neural network topology to capture the part of the network running 
on the systolic arrays in the cluster. 
For simulations we run the partition which has the most amount of computation such that we get the maximum runtime of all 
the parallel runs. 

In our experiments we examine two ways of partitioning *Output partitioning* and *Input channel partitioning*.
In each case the number of partitions are decided by the amount of under-utilization we encounter when the networks are 
mapped onto the equivalent array. 

1. **Output partitioning**: This scheme refers to partitioning the model such that different parts of the output feature map
(OFMAP) are generated on different logical systolic arrays.
To achieve this we resize the input feature map (IFMAP) dimensions in the topology csv file which is one of the inputs to 
SCALE-Sim. 

2. **Input channel partitioning**: In this scheme we generate different partial sums of a OFMAP pixel onto different logical 
systolic arrays. Logically this boild down to reducing the number of input channels in a given such array. 
As with the previous scheme, the topology csv input to SCALE-Sim is needed to be modified with the resulting number of channels.

## Replicating the Results

The following steps walks us through the steps to replicate our experiments from the FPL 2019 paper.

```
git clone gitlab@git.uwaterloo.ca:watcag-public/fpga-cascades-scalesim.git
TOP=`pwd`/fpga-cascades-scalesim
cd $TOP
```

### Pre-Requisites:
* python3
* scale-sim (provided in the folder)

You need the following packages: `tqdm, configparser, argparse`. Install as follows:
```
sudo -H pip3 install tqdm configparser argparse
```

### Input Files:

To run our simulations, you need to specify topology and configuration of the network.

The FPGA systolic array configuration is stored in `scalesim/scale.cfg`

Inspect this file as follows:
```
$ cat scalesim/scale.cfg
[general]
run_name = "run"

[architecture_presets]
ArrayHeight:	9
ArrayWidth:	960
IfmapSramSz:	30240
FilterSramSz:	4320
OfmapSramSz:	34560
IfmapOffset:	0
FilterOffset:	10000000
OfmapOffset:	20000000
Dataflow:	ws
```

The 'architecture_presets' of SCALE-Sim are as follows:
1. *ArrayHeight,
    ArrayWidth*: As the names suggest these are the height and width of the systolic array that SCALE-Sim needs to 
                 simulate. In our project, we model the FPGA DSP cascades as a 9x960 systolic array.
2. *IfmapSramSz, 
    FilterSramSz, 
    OfmapSramSZ*:   The are the logical SRAM partitions for storing the input feature map (IFMAP), convolution filters (Filter),
                    output feature maps (OFMAP) respectively. The unit of memory is kBytes.
3. *IfmapOffset,
    FilterOffset,
    OfmapOffset*: The offset values to the addresses to distinguish elements of the IFMAP, Filter, and OFMAP matrices respectively.
                  The values chosen here ensure that the elements of various matrices have distict address spaces.
4. *Dataflow*: Dataflow in a DNN accelerator signifies the mapping of compute and thus dictate the access pattern to the SRAMs 
             external memory. SCALE-Sim supports three dataflows; (i) Output Stationary indicated by 'os', (ii) Weight Stationary 
             indicated by 'ws', and (iii) Input Stationary indicated by 'is'. 
             In our work the compute mapping of the convolution engines follow the 'ws' dataflow, while the compute mapping in 
             mat-mul engines follow the 'is' dataflow.
             
In the example above we can see that the cascades are configured to work in Conv mode (indicated by 'ws' dataflow). 
The IFMAPs in this case are fed by part of URAM resulting in 252 kbit per processor. Since there are 960 processors in this configuration,
the total memory allocated for IFMAPs = 252 kbit * 960 / 8 Bytes = 30240 kBytes. The effective memory backing filter matrices is 36 kbits 
per processor. Generated OFMAPs are buffered by a dedicated URAM in the conv case, this resulting in a cumulative size of 34560 kB over 
the entire FPGA.

SCALE-Sim also needs a CNN topology file. For instance, you can look at AlphaGoZero in MLPerf as follows:
```
$ cat mlperf/AlphaGoZero.csv
Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,
Conv, 19, 19, 3, 3, 17, 256, 1,
Res_conv1, 19, 19, 3, 3, 256, 256, 1,
Res_conv2, 19, 19, 3, 3, 256, 256, 1,
ValueHead_conv, 19, 19, 1, 1, 256, 1, 1,
ValueHead_FC1, 1, 1, 1, 1, 361, 256, 1,
ValueHead_FC2, 1, 1, 1, 1, 256, 1, 1,
PolicyHead_Conv, 19, 19, 1, 1, 256, 2, 1,
PolicyHead_FC, 1, 1, 1, 1, 722, 362, 1,
```

This file captures the layer-wise hyperparameter for the given workload. As one can figure from the headers, the hyper-parameters 
are described as convolution matrix sizes such as the dimensions of the input feature map (IFMAP height, IFMAP Width, Channels), 
dimensions of filters (Filter height, Filter width, channels), number of filters (Num Filters), and strides. Fully connected layers
are described as convolutions where the input size is same as the filter size.

#### Create partitioned CSV for given MLPerf and given FPGA systolic array size

For this, you can pay attention to the following directory structure:
```
fpga-cascades-scalesim
├───mlperf             # MLPerf workload configuration --> original CSV files
└───scripts   
  └───topology_scripts # Scripts for partitioning the layers across the FPGA --> generate topology CSV files
```

This is performed by the `create_csv.py` file. This script takes the total number of cols in 
the equivalent arrays, and then computes the partitioned csv such that the utilization is maximum when run in scaled out configuration for each layer.  
Usage:
```
    create_csv.py [-h] [-c Number of columns] [-d Dataflow mode] [-t CNN topology] [-s Split mode] 
```
where,

- Number of cols: Number of processors
- Dataflow mode : Valid values are, '*ws*' for Conv mode, and '*is*' for Mat-Mul mode
- CNN topology  : Path to the unpartitioned file  
- Split mode    : Dimension to parallelise along. o: output feature map
                  (scaled out), c: along channels                                      
         
**Example,**
```
$python create_csv.py -c 960 -d is -t ../../mlperf/AlphaGoZero.csv -s c    
```

Running this command generates a file in named `AlphaGoZero_is_960.csv` in the directory `fpga-cascades-scalesim/mlperf/`.
The command instructs the script to generate partitioned csv file, by splitting along channels to improve the utilization of 480 processors
working the the mat-mul configuration. 

We look into the first layer of the original and the generated file to understand the operation. 

First layer in original file.
```
Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,
Conv, 19, 19, 3, 3, 17, 256, 1,
```

Note that the number of equivalent systolic arrays cols available are *960*, however this layer can use only *17x17* or *289* cols before partition.
This means that we are using less than a third of available resources. The utilization can be increased by partitioning the 
workload into atleast three whole parts in this case. 
Since the inputs instruct splitting along the channel dimension, we get the following hyperparameter configuration as output. 

First layer in generated file.
```
Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,
Conv,  19,  19,  3,  3, 6, 256,  1,
```

Notice that channels in the generated file = ceiling[17/3] = *6*

#### Partitioning all MLPerf CSVs + FPGA configurations in a batch

The script `create_max_util_topo.py` is a wrapper script around `create_csv.py`
generate partitioned CSV files for all MLPerf csv files corresponding to our workloads. The user can sepecify the split mode and the location of the directory.
Usage:
```
    create_max_util_topo.py [-h] [-s Split mode] [-p Topology directory]
```
where,

- Split mode            : Valid values are, '*o*' for Output partitioning, and '*c*' for Channel partitioning
- Topology directory    : Path to MLPerf folder with all networks

#### Create FPGA Systolic Array Configuration
The script '*create_cfg.py*' produces the SCALE-Sim configuration file given the number of 
processors, operating mode as dataflow, and path to the topology file. 
Usage:
```            
   create_cfg.py [-h] [-c Number of columns] [-d Dataflow mode] [-t CNN topology]
```
where,

- Number of cols: Number of processors
- Dataflow mode : Valid values are, '*ws*' for Conv mode, and '*is*' for Mat-Mul mode
- CNN topology  : Path to the unpartitioned file 

Example, 
```
$python create_cfg.py -c 480 -d ws -t ../../mlperf/AlphaGoZero.csv
```

Creates the file `scale.py` with the following contents:
```
[general]
run_name = "run"

[architecture_presets]
ArrayHeight:	9
ArrayWidth:	480
IfmapSramSz:	15120
FilterSramSz:	2160
OfmapSramSz:	17280
IfmapOffset:	0
FilterOffset:	10000000
OfmapOffset:	20000000
Dataflow:	ws

[network_presets]
TopologyCsvLoc:	"../../mlperf/AlphaGoZero.csv"
```

Note the changes in the SRAM sizes changing with the number of processors. 
For instance, the IFMAP SRAM size for conv units in `9x960` configuration we saw above was *30240* kB, which has now scaled to *15120* kB 
in proportion with the number of processors.
   
#### Launching runs
    
SCALE-Sim code is provided in the repo. We now demonstrate how the simulations can be run.

For this, you can pay attention to the following directory structure:
```
fpga-cascades-scalesim
├───scalesim            # Customized SCALE-Sim simulator for our framework
|  └───outputs          # Output of `scale.py` is stored here
└───bulk_scale_runs     # Scripts for running all MLPerf workloads, and all 
   └───scale_runs       # Output of `launch_all_max_util_runs.py` is stored here
```
 
#### Single Run
SCALE-Sim requires two input files, a csv file with the topology hyper-parameters, and a configuration 
file (`scale.cfg`) which captures the configuration of the systolic array it is simulating. The previous step shows how 
to generate both the files. `scale.cfg` is assumed to be placed in the same folder as `scale.py`.
    
You can execute the simulation as follows:

```
    pushd $TOP/scalesim
    python scale.py -t ../mlperf/AlphaGoZero.csv -c ./scale.cfg
```             

A sample output of the above command with for first two layers look like the following.       
Once you run SCALE-Sim, the outputs will be available in the directory `scalesim/outputs/`

```

====================================================
******************* SCALE SIM **********************
====================================================
Array Size: 	9x960
SRAM IFMAP: 	30240
SRAM Filter: 	4320
SRAM OFMAP: 	34560
CSV file path: 	../mlperf/AlphaGoZero.csv
Dataflow: 	Weight Stationary
====================================================

Commencing run for Conv
Generating traces and bw numbers
Generating DRAM traffic
Average utilization : 	26.666666666666654 %
Cycles for compute  : 	9571 cycles
DRAM IFMAP Read BW  : 	0.2938331896964474 Bytes/cycle
DRAM Filter Read BW : 	1.8753231829933927 Bytes/cycle
DRAM OFMAP Write BW : 	3.542277123431964 Bytes/cycle

Commencing run for Res_conv1
Generating traces and bw numbers
Generating DRAM traffic
Average utilization : 	26.66666666666674 %
Cycles for compute  : 	144128 cycles
DRAM IFMAP Read BW  : 	0.4390121087459444 Bytes/cycle
DRAM Filter Read BW : 	2.8018944558189913 Bytes/cycle
DRAM OFMAP Write BW : 	0.35145290700160087 Bytes/cycle
```       

*Explanation*: This is the standard output of SCALE-Sim. The following text provides a brief context and more information can 
be found in the [SCALE-Sim Repo](https://github.com/ARM-software/SCALE-Sim)

- Header: This part echoes the run configuration, like the workload, the array dimensions, and memory sizes
- Layer wise parameters: SCALE-Sim processes each layer eg. each line of the topology file at a time. 
                         per layer metrics are grouped together like below codeblock.
                         The reported metrics are:
                         
 - *Average utilization*: The utilization of the arrays averaged over time
 - *Cycles to compute*: Cycles to process the MAC operations in the layer assuming no stalls.
 - *DRAM IFMAP Read BW*: BW required in Bytes/cycle for fetching elements of IFMAP to ensure stall free operation 
 - *DRAM Filter Read BW*: BW required in Bytes/cycle for fetching elements of filter to ensure stall free operation 
 - *DRAM OFMAP Read BW*: BW required in Bytes/cycle for fetching elements of OFMAP to ensure stall free operation 
 

 ```
Commencing run for Conv
Generating traces and bw numbers
Generating DRAM traffic
Average utilization : 	26.666666666666654 %
Cycles for compute  : 	9571 cycles
DRAM IFMAP Read BW  : 	0.2938331896964474 Bytes/cycle
DRAM Filter Read BW : 	1.8753231829933927 Bytes/cycle
DRAM OFMAP Write BW : 	3.542277123431964 Bytes/cycle
```

#### Multiple runs
The fastest way to replicate our experiments is to launch multiple parallel scale-sim runs. 
This can be achieved using the scripts in the `bulk_scale_runs/` directory. The scripts in this directory fires
parallel runs for both mat-mul and conv configurations for array sizes ranging from 9x96 to 9x960, with the number of cols 
increasing in mulitples of 96

The steps are as follows:
    
- First create the partitioned directories using the '*create_max_util_topo.py*' script.
- Copy the generated topologies into a directory named '*max_util_topo*' in '\<top\>/bulk_scale_runs/'. 
-  Ensure that there are two sub-directories, 'ws' and 'is' within this directory. 
- Fire up the runs with the following command:

```  
    pushd $TOP/bulk_scale_runs
    python launch_all_max_util_runs.py
```

Example output: 
```
Started jobs 0 to 10 for FasterRCNN 
```

Note: Bulk jobs can take a few hours, please be patient.
   
#### Gather data

```
fpga-cascades-scalesim
└───scripts                     
  └───output_parse_scripts      # Analysis and post-processing of simulation data.
```

We also have a set of scripts to parse the generated data from the simulations. 

- `bulk_run_output_parser.py` takes the path to the scale_runs folder (this is the one with outputs of bulk runs) and 
creates an intermediate folder called `parsed_out` where it aggregates and stores the runs metrics for different array sizes
and dataflow. 
The data is categorized based on workloads and are stored in CSV files.
One can then used these CSV files as inputs and write more scripts for dataprocessing.

- `process_parsed.py`: This file contains methods to analyze the contents of the `parsed_out` file and create highler level datastructures 
for more refined examination. We encourage the users to use this file as a reference only to build their own methods to work with 
the parsed data.

## LICENSE

MIT License

Copyright (c) 2019 Anand Samajar, Tushar Krishna, Nachiket Kapre

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


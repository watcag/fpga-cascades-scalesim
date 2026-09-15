import math
import argparse
import os
import re

# Configure the parser
parser = argparse.ArgumentParser(description='CNN CSV Generator')
parser.add_argument('-c', metavar='Number of columns', type=int, default=960, help='the number of colummns in the systolic array')
parser.add_argument('-d', metavar='Dataflow mode', type=str, default="ws", help='choose between weight-stationary and input-stationary')
parser.add_argument('-t', metavar='CNN topology', type=str, default="mlperf/DeepSpeech2.csv", help='CSV file describing the CNN topology optimized for this particular systolic array size')
parser.add_argument('-s', metavar='Split mode', type=str, default='o', help='Dimension to parallelise along. o: output feature map (scaled out), c: along channels')
args = parser.parse_args()

num_cols = args.c
df = args.d
topo_raw = args.t
split_mode = args.s
#topo = re.sub('\.csv$', '', topo_raw)

topo1 = topo_raw.split('.')
topo ='.'.join(topo1[0:-1])

# Creates the partitioned topology for conv processors
# This function partitions the inputs along the channels.
# eg. if IFMAP is 224x224x100 and num partition is 10
#     the resulting IFMAP after partition is 224x224x10
#
# Note: We generate the configuration corresponding to slowest 
#       of the mix
# Parameters --
#   topo:       The topology filename
#   simd_mode:  1 = simd_mode off, 2 = simd_mode on 
#
def create_ws_topologies(topo, simd_mode=1):
    readfilename = topo + ".csv"
    readfile = open(readfilename,'r')

    writefilename = topo + "_ws_" + str(num_cols) + ".csv"
    writefile = open(writefilename, 'w')

    firstline = True

    # Parse the network file line by line
    for line in readfile:
        if firstline: 
            writefile.write(line)
            firstline = False

        elif not line == "":
            params = line.strip().split(',')
            
            ifmap_h = int(params[1].strip())
            filter_h = int(params[3].strip())
            channels = int(params[5].strip())
            num_filt = int(params[6].strip())
            stride   = int(params[7].strip())
        
            # Calcualte the number of partitions
            # The upper limit of partitions is the number of channels
            m_factor = math.ceil(num_cols / float(num_filt))
            m_factor = min(m_factor, channels)

            # Update the number of channels based on the number of partitions
            new_channels = int(math.ceil(channels / m_factor))
            #print(str(params[0].strip()) + ", " + str(m_factor))
    
            # If simd_mode=2 then split the IFMAP into two
            new_ifmap_h = int(math.ceil(ifmap_h/simd_mode))  
            if simd_mode == 2:
                new_ifmap_h += filter_h - stride
                new_ifmap_h = int(new_ifmap_h / stride)
                new_ifmap_h = max(new_ifmap_h, 1) # Trying to avoid a zero

            # Create a new line with the modified hyper-paramters and
            # write it to the writefile
            log = ""
            log += str(params[0]) +", "
            log += str(new_ifmap_h) + ", " 
            for i in range(2,5):
                log += str(params[i]) + ", "

            log += str(new_channels) + ", "
            log += str(num_filt) + ", "
            log += str(params[7]) + ",\n"
            writefile.write(log)

    readfile.close()
    writefile.close()


# Creates the partitioned topology for mat-mul processors
# This function partitions the inputs along the channels.
# eg. if IFMAP is 224x224x100 and num partition is 10
#     the resulting IFMAP after partition is 224x224x10
#
# Note: We generate the configuration corresponding to slowest 
#       of the mix
# Parameters --
#   topo:       The topology filename
#
def create_is_topologies(topo):
    readfilename = topo + ".csv"
    readfile = open(readfilename,'r')

    writefilename = topo + "_is_" + str(num_cols) + ".csv"
    writefile = open(writefilename, 'w')

    firstline = True

    # Parse the topology file line by line
    for line in readfile:
        if firstline: 
            writefile.write(line)
            firstline = False

        elif not line == "":
            params = line.strip().split(',')
            
            ifmap_h  = int(params[1].strip())
            ifmap_w  = int(params[2].strip())
            filter_h = int(params[3].strip())
            filter_w = int(params[4].strip())
            channels = int(params[5].strip())
            num_filt = int(params[6].strip())
            strides  = int(params[7].strip())
        
            eh = math.ceil((ifmap_h - filter_h + strides) / strides)
            ew = math.ceil((ifmap_w - filter_w + strides) / strides)
            e2 = eh * ew
    
            # Calculate the number of partitions and 
            # limit it by the number of channels
            m_factor = max(math.floor(num_cols / e2),1)
            m_factor = min(m_factor, channels)
    
            # Update the new number of channels
            new_channels = int(math.ceil(channels / m_factor))

            log = ""
            for i in range(5):
                log += str(params[i]) + ", "

            log += str(new_channels) + ", "
            log += str(num_filt) + ", "
            log += str(params[7]) + ",\n"

            writefile.write(log)

    readfile.close()
    writefile.close()


# The following code is for scaled-out CSV generation
# 
# Creates the partitioned topology for conv processors
# This function partitions the inputs along the OFMAP pixel.
# Essentially this is exploiting parallelism by model-partitioning
# eg. if IFMAP is 102x102x3 and num partition is 10
#     the resulting IFMAP after partition is 12x102x3
#
# Note: We generate the configuration corresponding to slowest 
#       of the mix
# Parameters --
#   topo:       The topology filename
#   simd_mode:  1 = simd_mode off, 2 = simd_mode on 
#
def create_scaled_out_ws_topologies(topo, simd_mode=1):
    readfilename = topo + ".csv"
    readfile = open(readfilename,'r')

    writefilename = topo + "_ws_" + str(num_cols) + ".csv"
    writefile = open(writefilename, 'w')

    firstline = True

    for line in readfile:
        if firstline: 
            writefile.write(line)
            firstline = False

        elif not line == "":
            params = line.strip().split(',')
            
            ifmap_h = int(params[1].strip())
            ifmap_w = int(params[2].strip())
            filter_h = int(params[3].strip())
            filter_w = int(params[4].strip())
            channels = int(params[5].strip())
            num_filt = int(params[6].strip())
            stride   = int(params[7].strip())

            # Calculate the number of ofmap pixel per filter
            e_h = math.ceil((ifmap_h - filter_h + stride)/stride)
            e_w = math.ceil((ifmap_w - filter_w + stride)/stride)
            e2  = e_h * e_w     # Number of ofmap px per filter (single output channel)

            # Calculation of the scaling factor
            num_scaled_out_arr = math.floor((num_cols * simd_mode) / num_filt)
            num_scaled_out_arr = max(num_scaled_out_arr, 1)
            scaled_out_ofmap_per_col = e2 / num_scaled_out_arr      # As we are scaling the number of 
                                                                    # px to be generated by a single col

            # SCALE-Sim has a limitation that the IFMAP can only be rectangular
            # Thus we need to allocate OFMAP px accordingly on cols to meet this req
            # Note: This script generates the topology for the worst case allocation
            #       ie. in case there is uneven distribution, the generated topology
            #       corresponds to the one with most work. Thus the numbers below are
            #       conservative
            if scaled_out_ofmap_per_col < 1:
                scaled_out_ofmap_per_col = 1        # Will generate at least one px per col

                new_ifmap_h = filter_h
                new_ifmap_w = filter_w

            elif scaled_out_ofmap_per_col <= e_w:
                scaled_out_ofmap_per_col = math.ceil(scaled_out_ofmap_per_col)  # Make it an integer, and don't underestimate compute

                new_ifmap_w = (scaled_out_ofmap_per_col - 1) * stride + filter_w    # To reason about -1 think about when num_ofmap_px == 1
                new_ifmap_h = filter_h

            else:   # This is essenetially scaled_out_ofmap_per_col > e_w
                scaled_out_ofmap_per_col = math.ceil(scaled_out_ofmap_per_col)  # Rounding could have worked; but we are conservative here

                new_ifmap_w = ifmap_w   # Original width
                new_ifmap_h = (scaled_out_ofmap_per_col - 1) * stride + filter_h

            log = ""
            log += str(params[0]) +", "
            log += str(new_ifmap_h) + ", " 
            log += str(new_ifmap_w) + ", " 
            for i in range(3,8):
                log += str(params[i]) + ", "

            log += "\n"
            writefile.write(log)
        
    readfile.close()
    writefile.close()

# Creates the partitioned topology for mat-mul processors
# This function partitions the inputs along the OFMAP pixel.
# Essentially this is exploiting parallelism by model-partitioning
# eg. if IFMAP is 102x102x3 and num partition is 10
#     the resulting IFMAP after partition is 12x102x3
#
# Note: We generate the configuration corresponding to slowest 
#       of the mix
# Parameters --
#   topo:       The topology filename
#
def create_scaled_out_is_topologies(topo):
    readfilename = topo + ".csv"
    readfile = open(readfilename,'r')

    writefilename = topo + "_is_" + str(num_cols) + ".csv"
    writefile = open(writefilename, 'w')

    firstline = True

    for line in readfile:
        if firstline: 
            writefile.write(line)
            firstline = False

        elif not line == "":
            params = line.strip().split(',')
            
            ifmap_h = int(params[1].strip())
            ifmap_w = int(params[2].strip())
            filter_h = int(params[3].strip())
            filter_w = int(params[4].strip())
            channels = int(params[5].strip())
            num_filt = int(params[6].strip())
            stride   = int(params[7].strip())

            # Calculate the number of ofmap pixel per filter
            e_h = math.ceil((ifmap_h - filter_h + stride)/stride)
            e_w = math.ceil((ifmap_w - filter_w + stride)/stride)
            e2  = e_h * e_w     # Number of ofmap px per filter (single output channel)

            # Calculation of the scaling factor
            num_scaled_out_arr = math.floor((num_cols) / e2)
            num_scaled_out_arr = max(num_scaled_out_arr, 1)
            scaled_out_filter_per_col = math.ceil(num_filt/ num_scaled_out_arr)

            new_num_filt = scaled_out_filter_per_col

            log = ""
            for i in range(0,6):
                log += str(params[i]) +", "
            log += str(new_num_filt) + ", " 
            log += str(params[7]) + ", " 

            log += "\n"
            writefile.write(log)
        
    readfile.close()
    writefile.close()

def run_this():
    if df == 'ws':
        if split_mode == 'c':
            create_ws_topologies(topo, simd_mode=2)
        else:
            create_scaled_out_ws_topologies(topo, simd_mode=2)

    elif df == 'is':
        if split_mode == 'c':
            create_is_topologies(topo)
        else:
            create_scaled_out_is_topologies(topo)

if __name__ == '__main__':
    run_this()

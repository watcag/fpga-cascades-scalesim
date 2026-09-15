import os
import argparse
import math

parser = argparse.ArgumentParser(description='Configuration Generator')
parser.add_argument('-c', metavar='Number of columns', type=int, default=960, help='the number of colummns in the systolic array')
parser.add_argument('-d', metavar='Dataflow mode', type=str, default="ws", help='choose between weight-stationary and input-stationary')
parser.add_argument('-t', metavar='CNN topology', type=str, default="mlperf/DeepSpeech2_ws_960.csv", help='CSV file describing the CNN topology optimized for this particular systolic array size')
args = parser.parse_args()

num_cols = args.c
df = args.d
topo = args.t

weight_mem_per_proc= [36, 288]      # Conv: 36 kbit, Mat Mul: 288 kbit 
input_mem_per_proc= [252,162]       # Conv: 288 - 36 kbit, Mat Mul:18 * 9 kbit
output_mem_per_proc = [288, 18]     # Conv: One URAM, Mat Mul: one BRAM

def generate_config(
                    arrh = 1,
                    arrw = 1,
                    ifmap_sram  = 512,
                    filter_sram = 512,
                    ofmap_sram  = 256,
                    ifmap_offset = 0,
                    filter_offset = 10000000,
                    ofmap_offset = 20000000,
                    dataflow = 'os',
                    topology = 'test.csv'
        ):

    config_headers = ["[general]", "[architecture_presets]", "[network_presets]"]

    outfile = open("scale.cfg", 'w')

    line = config_headers[0] + "\n"
    outfile.write(line)

    line = "run_name = \"run\"\n"
    outfile.write(line)

    line = "\n" + config_headers[1] + "\n"
    outfile.write(line)

    line = "ArrayHeight:\t" + str(arrh) + "\n" 
    outfile.write(line)

    line = "ArrayWidth:\t" + str(arrw) + "\n" 
    outfile.write(line)

    line = "IfmapSramSz:\t" + str(ifmap_sram) + "\n" 
    outfile.write(line)

    line = "FilterSramSz:\t" + str(filter_sram) + "\n" 
    outfile.write(line)

    line = "OfmapSramSz:\t" + str(ofmap_sram) + "\n" 
    outfile.write(line)

    line = "IfmapOffset:\t" + str(ifmap_offset) + "\n" 
    outfile.write(line)

    line = "FilterOffset:\t" + str(filter_offset) + "\n" 
    outfile.write(line)

    line = "OfmapOffset:\t" + str(ofmap_offset) + "\n" 
    outfile.write(line)

    line = "Dataflow:\t" + dataflow + "\n" 
    outfile.write(line)

    line = "\n" + config_headers[2] + "\n"
    outfile.write(line)

    line = "TopologyCsvLoc:\t\"" + topology + "\"\n"
    outfile.write(line)

    outfile.close()

def create_cfg_files():
    first = True
    factor = 1

    if(df=="ws"):
        df_idx=0;
    else:
        df_idx=1;

    weight_mem = int(weight_mem_per_proc[df_idx] * num_cols / 8)  # div 8 converts to KB
    input_mem = int(input_mem_per_proc[df_idx] * num_cols / 8)
    output_mem = int(output_mem_per_proc[df_idx] * num_cols / 8)   

    num_rows=9
    generate_config(
         arrh=num_rows,
         arrw=num_cols,
         ifmap_sram=input_mem,
         filter_sram=weight_mem,
         ofmap_sram=output_mem,
         dataflow=df,
         topology=topo
    )

create_cfg_files()

import os
import argparse

# Global variable lists
# These are the SCALE-Sim dataflow and MLPerf + some extra toplogy file names
df_list = ['ws', 'is']
topo_list = [
                'AlphaGoZero',
                'DeepSpeech2',
                'FasterRCNN',
                'NCF_recommendation',
                'Resnet50',
                'Sentimental_seqCNN',
                'Googlenet',
                'vgg16',
                'alexnet'
            ]

# Wrapper to create the partitioned topologies and manage directories
def create_max_util_topo(split_mode='o', topodir="../../mlperf/"):

    # Top output directory
    cmd = 'mkdir ./max_util_topo'
    os.system(cmd)

    # Run this for each dataflow in the list
    for df in df_list:

    # Loop over from col size = 96 to 960
        for i in range(1,11):
            num_proc = 96 * i

            # For each topology run the division script
            for topo in topo_list:
                cmd = "python create_csv.py "
                cmd += "-c " + str(num_proc)
                cmd += " -d " + df
                cmd += " -s " + split_mode
                cmd += " -t " + topodir + "/" + topo + ".csv"


                #print(cmd + "\n")
                os.system(cmd)

                # Move the generated file to the directory we created
                name = topo + "_" + df + "_" + str(num_proc) + ".csv"
                print(name)
                cmd = "mv " + topodir + "/" + topo + "_" + df + "_" + str(num_proc) + ".csv ./max_util_topo/"
                os.system(cmd)

        #cmd = 'cp -r ./max_util_topo ../mlperf/'
        #os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bulk CNN CSV Generator')
    parser.add_argument('-s', metavar='Split mode', type=str, default='o',
                        help='Dimension to parallelise along. o: output feature map (scaled out), c: along channels')
    parser.add_argument('-p', metavar='Topology directory', type=str, default='../../mlperf',
                        help="Location to the topology directory")
    args = parser.parse_args()

    split_mode = args.s
    topo_dir = args.p
    create_max_util_topo(split_mode=split_mode, topodir= topo_dir)


import os
import time
import argparse 
import configparser as cp
import run_nets as r


class scale:
    def __init__(self, sweep = False, save = False):
        self.sweep = sweep 
        self.save_space = save

        self.config_file = ''
        self.topology_file = ''
        self.silent = False

    def set_params(self, config_file='./scale.cfg', topo_file='', silent_flag=False):
        self.config_file = config_file
        self.topology_file = topo_file
        self.silent = silent_flag

    def parse_config(self):
        general = 'general'
        arch_sec = 'architecture_presets'
        net_sec  = 'network_presets'
        config_filename = self.config_file
    
        config = cp.ConfigParser()
        config.read(config_filename)
    
        ## Read the run name
        self.run_name = config.get(general, 'run_name')

        ## Read the architecture_presets
        ## Array height min, max
        ar_h = config.get(arch_sec, 'ArrayHeight').split(',')
        self.ar_h_min = ar_h[0].strip()
    
        if len(ar_h) > 1:
            self.ar_h_max = ar_h[1].strip()
        ##print("Min: " + ar_h_min + " Max: " + ar_h_max)
    
        ## Array width min, max
        ar_w = config.get(arch_sec, 'ArrayWidth').split(',')
        self.ar_w_min = ar_w[0].strip()
        
        if len(ar_w) > 1:
            self.ar_w_max = ar_w[1].strip()
    
        ## IFMAP SRAM buffer min, max
        ifmap_sram = config.get(arch_sec, 'IfmapSramSz').split(',')
        self.isram_min = ifmap_sram[0].strip()
        
        if len(ifmap_sram) > 1:
            self.isram_max = ifmap_sram[1].strip()
       
    
        ## FILTER SRAM buffer min, max
        filter_sram = config.get(arch_sec, 'FilterSramSz').split(',')
        self.fsram_min = filter_sram[0].strip()
        
        if len(filter_sram) > 1:
            self.fsram_max = filter_sram[1].strip()
        
        
        ## OFMAP SRAM buffer min, max
        ofmap_sram = config.get(arch_sec, 'OfmapSramSz').split(',')
        self.osram_min = ofmap_sram[0].strip()
        
        if len(ofmap_sram) > 1:
            self.osram_max = ofmap_sram[1].strip()
    
        self.dataflow= config.get(arch_sec, 'Dataflow')
    
        ifmap_offset = config.get(arch_sec, 'IfmapOffset')
        self.ifmap_offset = int(ifmap_offset.strip())

        filter_offset = config.get(arch_sec, 'FilterOffset')
        self.filter_offset = int(filter_offset.strip())

        ofmap_offset = config.get(arch_sec, 'OfmapOffset')
        self.ofmap_offset = int(ofmap_offset.strip())

        ## Read network_presets
        ## For now that is just the topology csv filename
        if self.topology_file == '':
            topology_file = config.get(net_sec, 'TopologyCsvLoc')
            self.topology_file = topology_file.split('"')[1]     #Config reads the quotes as well


    def run_scale(self):
        self.parse_config()

        if self.sweep == False:
            self.run_once()
        else:
            self.run_sweep()


    def run_once(self):

        df_string = "Output Stationary"
        if self.dataflow == 'ws':
            df_string = "Weight Stationary"
        elif self.dataflow == 'is':
            df_string = "Input Stationary"

        if not self.silent:
            print("====================================================")
            print("******************* SCALE SIM **********************")
            print("====================================================")
            print("Array Size: \t" + str(self.ar_h_min) + "x" + str(self.ar_w_min))
            print("SRAM IFMAP: \t" + str(self.isram_min))
            print("SRAM Filter: \t" + str(self.fsram_min))
            print("SRAM OFMAP: \t" + str(self.osram_min))
            print("CSV file path: \t" + self.topology_file)
            print("Dataflow: \t" + df_string)
            print("====================================================")

        net_name = self.topology_file.split('/')[-1].split('.')[0]
        ##print("Net name = " + net_name)
        offset_list = [self.ifmap_offset, self.filter_offset, self.ofmap_offset]
        
        r.run_net(  silent=self.silent,
                    ifmap_sram_size  = int(self.isram_min),
                    filter_sram_size = int(self.fsram_min),
                    ofmap_sram_size  = int(self.osram_min),
                    array_h = int(self.ar_h_min),
                    array_w = int(self.ar_w_min),
                    net_name = net_name,
                    data_flow = self.dataflow,
                    topology_file = self.topology_file,
                    offset_list = offset_list
                )
        self.cleanup()
        #print("************ SCALE SIM Run Complete ****************") 


    def cleanup(self):
        if not os.path.exists("./outputs/"):
            os.system("mkdir ./outputs")
        
        net_name = self.topology_file.split('/')[-1].split('.')[0]

        path = "./output/scale_out"
        if self.run_name == "":
            path = "./outputs/" + net_name +"_"+ self.dataflow
        else: 
            path = "./outputs/" + self.run_name

        if not os.path.exists(path):
            os.system("mkdir " + path)
        else:
            t = time.time()
            new_path= path + "_" + str(t)
            os.system("mv " + path + " " + new_path)
            os.system("mkdir " + path)


        cmd = "mv *.csv " + path
        os.system(cmd)

        cmd = "mkdir " + path +"/layer_wise"
        os.system(cmd)

        cmd = "mv " + path +"/*sram* " + path +"/layer_wise"
        os.system(cmd)

        cmd = "mv " + path +"/*dram* " + path +"/layer_wise"
        os.system(cmd)

        if self.save_space == True: 
            cmd = "rm -rf " + path +"/layer_wise"
            os.system(cmd)


    def run_sweep(self):

        all_data_flow_list = ['os', 'ws', 'is']
        all_arr_dim_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        all_sram_sz_list = [256, 512, 1024]

        data_flow_list = all_data_flow_list[1:3]
        arr_h_list = all_arr_dim_list[4:5]
        arr_w_list = all_arr_dim_list[3:8]
        #arr_w_list = list(reversed(arr_h_list))
        
        net_name = self.topology_file.split('/')[-1].split('.')[0]
        for df in data_flow_list:
            self.dataflow = df

            for i in range(len(arr_h_list)):
                self.ar_h_min = arr_h_list[i]
                self.ar_w_min = arr_w_list[i]

                self.run_name = net_name + "_" + df + "_" + str(self.ar_h_min) + "x" + str(self.ar_w_min)

                self.run_once()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Systolic CNN AcceLErator SIMulator')
    parser.add_argument('-c', metavar='SCALE-Sim configuration file', type=str, default='./scale.cfg',
                        help='path + name of the scale configuration file')
    parser.add_argument('-t', metavar='Topology CSV file', type=str, default='',
                        help='path + name of the topology CSV file')
    parser.add_argument('-v', metavar='Verbosity', type=str, default='True',
                        help='Turn screen outptus on and off')
    
    args = parser.parse_args()

    config_file = args.c
    topo_file = args.t
    verbose = args.v

    if verbose == 'True' or verbose == 'T' or verbose == '1':
        silent = False          # This is correct 
    else:
        silent = True

    s = scale(save = False, sweep = False)
    s.set_params(config_file=config_file, topo_file=topo_file, silent_flag=silent)
    s.run_scale()

    

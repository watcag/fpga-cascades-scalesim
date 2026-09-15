import os
import argparse

def parse_cycle_logs(list_dirs, config_list, topo_list, df):
    topo_name = topo_list[0].split('_')[0]  # Otherwise the col names and df are appended 
    #topo_name = topo_list[0]
    
    # Handle exceptions here
    except_list = ['NCF', 'Sentimental']
    replace_list = ['NCF_recommendation', 'Sentimental_seqCNN']

    if topo_name in except_list:
        idx = except_list.index(topo_name)
        topo_name = replace_list[idx]

    cycles_filename = topo_name + "_" + df +"_cycles.csv"
    utils_filename  = topo_name + "_" + df +"_utils.csv"

    cycles_file = open(cycles_filename, 'w')
    utils_file  = open(utils_filename, 'w')
    
    new_log = True
    
    for dir_name, config, topo in zip(list_dirs, config_list, topo_list):
        path = dir_name
        print(path)
    
        r_filename = path + topo + "_cycles.csv"
        read_file = open(r_filename, 'r')

        cycles = config + ","  
        utils =  config + ","
        layer_name = df + ","

        first = True
    
        for line in read_file:
            if first: 
                first = False
            else:
                elems = line.split(',')
                layer_name += elems[0].strip() + ","
                cycles += elems[1].strip()+ "," 
                utils  += elems[2].strip()+ "," 

        read_file.close()
        if new_log:
            cycles_file.write(layer_name + "\n")
            utils_file.write(layer_name + "\n")
            new_log = False
            
        cycles_file.write(cycles + "\n")
        utils_file.write(utils + "\n")

    cycles_file.close()
    utils_file.close()


def parse_bw_logs(list_dirs, config_list, topo_list, df):
    topo_name = topo_list[0].split('_')[0]
    #topo_name = topo_list[0]
    
    # Handle exceptions here
    except_list = ['NCF', 'Sentimental']
    replace_list = ['NCF_recommendation', 'Sentimental_seqCNN']

    if topo_name in except_list:
        idx = except_list.index(topo_name)
        topo_name = replace_list[idx]
    
    sram_read_filename   = topo_name + "_" + df +"_sram_rd_bw.csv"
    sram_write_filename  = topo_name + "_" + df +"_sram_wr_bw.csv"
    dram_ifmap_filename  = topo_name + "_" + df +"_dram_ifmap_rd_bw.csv"
    dram_filter_filename = topo_name + "_" + df +"_dram_filter_rd_bw.csv"
    dram_write_filename  = topo_name + "_" + df +"_dram_wr_bw.csv"

    sram_rd_file  = open(sram_read_filename, 'w')
    sram_wr_file  = open(sram_write_filename, 'w')
    dram_wr_file  = open(dram_write_filename, 'w')
    dram_ifmap_file  = open(dram_ifmap_filename, 'w')
    dram_filter_file  = open(dram_filter_filename, 'w')

    new_log = True

    for dir_name, config, topo in zip(list_dirs, config_list, topo_list):
        path = dir_name
        print(path)
    
        r_filename = path + topo + "_avg_bw.csv"
        read_file = open(r_filename, 'r')

        dram_ifmap  = config + ',' 
        dram_filter = config + ',' 
        dram_write  = config + ',' 
        sram_read   = config + ',' 
        sram_write  = config + ',' 
        layer_name  = df + ',' 
        
        first = True
    
        for line in read_file:
            if first: 
                first = False
            else:
                elems = line.split(',')
                layer_name += elems[3].strip() + ","
                dram_ifmap  += elems[4].strip()+ "," 
                dram_filter += elems[5].strip()+ "," 
                dram_write  += elems[6].strip()+ "," 
                sram_read   += elems[7].strip()+ "," 
                sram_write  += elems[8].strip()+ "," 

        read_file.close()
        if new_log:
            sram_rd_file.write(layer_name + "\n")
            sram_wr_file.write(layer_name + "\n")
            dram_wr_file.write(layer_name + "\n")
            dram_ifmap_file.write(layer_name + "\n")
            dram_filter_file.write(layer_name + "\n")
            new_log = False
        
        sram_rd_file.write(sram_read + "\n")
        sram_wr_file.write(sram_write + "\n")
        dram_wr_file.write(dram_write + "\n")
        dram_ifmap_file.write(dram_ifmap + "\n")
        dram_filter_file.write(dram_filter + "\n")

    sram_rd_file.close()
    sram_wr_file.close()
    dram_wr_file.close()
    dram_ifmap_file.close()
    dram_filter_file.close()


def parse_wrap(topo, runs_dir="./scale_runs"):
    
    path_list = []

    #prefix = "./scale_runs/"
    prefix = runs_dir + "/"

    arrh = 9
    min_arrw = 96

    dim_list = [x for x in range(1,11)]
    #dim_list += [20, 50]
    
    for df in ['ws','is']:
    
        path_list = []
        config_list = []
        topo_list = []
        for i in dim_list:
            arrw = int (min_arrw * i)
            config = str(arrh) + "x" + str(arrw)
            path = prefix + str(arrh) + "x" + str(arrw) + "_" + df
            path += "/outputs/" + topo + "_" + str(arrh) + "x" + str(arrw) + "_" + df + "/"
            path_list.append(path)
            config_list.append(config)

            topo_this = topo
            #topo_this += "_" + str(arrw) + "_" + df
            topo_this += "_" + df + "_" + str(arrw)
            topo_list.append(topo_this)
        
        print(path_list)
        print(topo_list)
        print(" ")

        parse_cycle_logs(path_list, config_list, topo_list, df)
        parse_bw_logs(path_list, config_list, topo_list, df)
        

topo_list = [
                "AlphaGoZero",
                "DeepSpeech2",                 
                "FasterRCNN",                 
                "Googlenet",
                "NCF_recommendation",                 
                "Resnet50",                 
                "Sentimental_seqCNN"
            ]


def parse_from_output_dir(run_dir="./scale_runs"):
    #topo_list = ["vgg16"]
    topo_list_local = topo_list
    print(topo_list_local)

    cmd = "mkdir parsed_out"
    os.system(cmd)

    for topo in  topo_list_local:
        cmd = "mkdir ./parsed_out/" + topo
        os.system(cmd)
        parse_wrap(topo, run_dir)
        
        cmd = "mv " + topo + "*.csv ./parsed_out/" + topo
        os.system(cmd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parse results for bulk runs')
    parser.add_argument('-r', metavar='Run output directory', default='../../bulk_scale_runs/scale_runs',
                        type=str, help='Path to the run directory')
    args = parser.parse_args()

    run_dir = args.r
    parse_from_output_dir(run_dir)

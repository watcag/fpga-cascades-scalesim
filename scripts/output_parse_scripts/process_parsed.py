import os

class process_parsed(object):
    def __init__(self):
        self.mlperf_topology_list = [
                                "AlphaGoZero",
                                "DeepSpeech2",
                                "FasterRCNN",
                                "NCF_recommendation",
                                "Resnet50",
                                "Sentimental_seqCNN",
                                #"Transformer_short",
                                "Googlenet"
                                ]
        self.conv_topo_list = []
        self.mult = [[2, 144, 144, 144, 144, 144, 145, 12, 12],
                     [2, 2, 1, 1, 1, 1]
                ]

        self.top_path =  os.path.dirname(os.path.realpath(__file__))

        #
        self.current_topology = ""
        self.matmul_freq = 650 * 10**6
        self.conv_freq = 650 * 10**6
        self.topo_read_flag = False
        self.topo_ws_runtime_arr = []
        self.topo_is_runtime_arr = []

        #
        self.topo_min_runtime_layerwise_arr = []
        self.topo_min_runtime_total_arr = []
        self.topo_min_runtime_total_ms_arr = []
        self.min_runtime_done_flag = False

        #
        self.cycles_conv_arr = []
        self.cycles_matmul_arr = []

        #
        self.target_datapath_top = []
    # End of __init__

    def reset_all_data(self):
        self.topo_read_flag = False
        self.topo_ws_runtime_arr = []
        self.topo_is_runtime_arr = []

        self.topo_min_runtime_total_arr = []
        self.topo_min_runtime_total_ms_arr = []
        self.topo_min_runtime_layerwise_arr = []
        self.min_runtime_done_flag = False
        #
        self.cycles_conv_arr = []
        self.cycles_matmul_arr = []
        self.target_datapath_top = []
    # End of reset_all_data

    def set_current_topo(self, topo):
        self.current_topology = topo
        self.reset_all_data()
    # End of set_current_topo

    def read_parsed_topo_runtimes(self):
        path = self.top_path + "/parsed_out/" + str(self.current_topology)

        # Read the is file
        filename = path + "/" + str(self.current_topology) + "_is_cycles.csv"
        f = open(filename, 'r')

        first = True
        self.topo_is_runtime_arr = []
        for row in f:
            row = row.strip()
            if first or row == "":
                first = False

            else:
                entries = row.split(',')
                elems = [int(x.strip()) for x in entries[1:-1]]
                self.topo_is_runtime_arr.append(elems)

        f.close()

        # Read the ws file
        filename = path + "/" + str(self.current_topology) + "_ws_cycles.csv"
        f = open(filename, 'r')

        first = True
        self.topo_ws_runtime_arr = []
        for row in f:
            row = row.strip()
            if first or row == "":
                first = False

            else:
                entries = row.split(',')
                elems = [int(x.strip()) for x in entries[1:-1]]
                self.topo_ws_runtime_arr.append(elems)
        f.close()

        self.topo_read_flag = True
    # End of read_parsed_topo_runtimes

    def compute_partioned_runtimes(self):
        if not self.topo_read_flag:
            self.read_parsed_topo_runtimes()

        self.topo_min_runtime_layerwise_arr = []
        self.topo_min_runtime_total_arr = []

        for i in range(9):
            idx_ws = i
            #idx_is = 9 - i
            idx_is = 8 - i

            ws_row = self.topo_ws_runtime_arr[idx_ws]
            is_row = self.topo_is_runtime_arr[idx_is]

            ms_runtime = 0
            cycles_runtime = 0
            cycles_conv =0
            cycles_matmul = 0
            this_layer_cycles = 0
            idx = 0
            this_layer_cycles_arr = []
            target_datapath_arr = []
            for x,y in zip(ws_row, is_row):
                this_layer_cycles = 0
                if x < y:
                    if self.current_topology == 'NCF_recommendation_short':
                        x = x * self.mult[1][idx]
                    elif self.current_topology == "Transformer_short":
                        x = x * self.mult[0][idx]

                    this_layer_cycles = x
                    cycles_runtime += x
                    cycles_conv += x
                    ms_runtime += 10**3 * (x/self.conv_freq)
                    dp = 'conv'
                else:
                    if self.current_topology == 'NCF_recommendation_short':
                        y = y * self.mult[1][idx]
                    elif self.current_topology == "Transformer_short":
                        y = y * self.mult[0][idx]

                    this_layer_cycles = y
                    cycles_runtime += y
                    cycles_matmul += y
                    ms_runtime += 10**3 * (y/self.matmul_freq)
                    dp  = 'matmul'

                this_layer_cycles_arr.append(this_layer_cycles)
                target_datapath_arr.append(dp)

            self.target_datapath_top.append(target_datapath_arr)
            self.cycles_conv_arr.append(cycles_conv)
            self.cycles_matmul_arr.append(cycles_matmul)
            self.topo_min_runtime_layerwise_arr.append(this_layer_cycles_arr)
            self.topo_min_runtime_total_arr.append(cycles_runtime)
            self.topo_min_runtime_total_ms_arr.append(ms_runtime)
            idx += 1

        self.min_runtime_done_flag = True
    # End of compute_partitioned_runtimes

    def save_partitioned_runtimes(self, filename="", verbose=True):
        if not self.min_runtime_done_flag:
            self.compute_partioned_runtimes()

        if filename=="":
            outfilename = self.top_path  + "/" + str(self.current_topology) + "_partitioned_runtime.csv"
        else:
            outfilename = self.top_path  + "/" + filename

        fo = open(outfilename,'w')

        header =  ', cycles, milliseconds,'
        if verbose: print(header)
        header += "\n"
        fo.write(header)

        for i in range(len(self.topo_min_runtime_total_ms_arr)):
            conv_part = int((i+1) * 10)
            matmul_part = int((9-i) * 10)

            conf = str(conv_part) + ":" + str(matmul_part)

            entry = ",".join([conf, str(self.topo_min_runtime_total_arr[i]), str(self.topo_min_runtime_total_ms_arr[i])])
            if verbose: print(entry)
            entry += ",\n"
            fo.write(entry)

        fo.close()

    def print_min_partitioned_runtime(self, layer_wise=False):

        min_runtime = min(self.topo_min_runtime_total_ms_arr)
        idx = self.topo_min_runtime_total_ms_arr.index(min_runtime)

        c  = int((idx+1) * 10)
        m  = int((9-idx) * 10)
        conf = str(c) + ":"  +str(m)

        print(conf + ", " + str(min_runtime))

        if layer_wise:
            filename = self.top_path + "/" + self.current_topology + "_min_cycles_layer_wise.csv"
            f = open(filename, 'w')

            layer_wise_arr = self.topo_min_runtime_layerwise_arr[idx]
            dp_arr = self.target_datapath_top[idx]
            for num, dp in zip(layer_wise_arr, dp_arr):
                entry = dp + ", " + str(num)
                print(entry)
                entry += ",\n"
                f.write(entry)

            f.close()


    def print_latex_table_min_partitioned_runtime(self, outfilename="min_runtime.tex"):

        f = open(outfilename, 'w')
        preamble = "\\begin{table}[]\n\\begin{tabular}{lllll}\n"

        header = "Topology"
        header += " & " + "Conv : Matmul"
        header += " & " + "Cycles conv"
        header += " & " + "Cycles matmul"
        header += " & " + "Time in ms"
        header += " \\\ \n"

        end = "\\end{tabular}\n\\end{table}\n"

        f.write(preamble)
        f.write(header)

        for topo in self.mlperf_topology_list:
            self.set_current_topo(topo)
            self.save_partitioned_runtimes()
            min_runtime = min(self.topo_min_runtime_total_ms_arr)
            idx = self.topo_min_runtime_total_ms_arr.index(min_runtime)

            c  = int((idx+1) * 10)
            m  = int((9-idx) * 10)
            conf = str(c) + ":"  +str(m)

            cycles_conv = self.cycles_conv_arr[idx]
            cycles_matmul = self.cycles_matmul_arr[idx]

            entry = topo
            entry += " & " + conf
            entry += " & " + str(cycles_conv)
            entry += " & " + str(cycles_matmul)
            entry += " & " + str(min_runtime)
            entry += " \\\ \n"
            f.write(entry)

        f.write(end)
        f.close()
    # End of print_latex_table_min_partitioned_runtime

if __name__ == "__main__":
    pp = process_parsed()
    #pp.print_latex_table_min_partitioned_runtime()
    for topo in pp.mlperf_topology_list:
        print(topo + ", ", end='')
        pp.set_current_topo(topo)
        pp.save_partitioned_runtimes(verbose=True)
        pp.print_min_partitioned_runtime(layer_wise=True)

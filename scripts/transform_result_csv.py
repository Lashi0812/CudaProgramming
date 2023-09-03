from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import pandas as pd 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_file")
    parser.add_argument("--group_keys",nargs="+",help='Allowed values ["ID","Kernel Name","Metric Name","Metric Unit","Metric Value"]',
                        default="ID")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_file,
                     usecols=["ID","Kernel Name","Metric Name","Metric Unit","Metric Value"])
    groups = df.groupby(args.group_keys)
    
    final_dict = defaultdict(list)
    index_list = []
    
    for group_key,group in groups:
        index_list.append(group_key)
        for name,unit,val in zip(group["Metric Name"].values,group["Metric Unit"].values,group["Metric Value"].values):
            if unit == "nsecond":
                val = val * 1.E-6
                unit = "millisec"
            if unit == "byte/second":
                val = val / (1024 **3)
                unit = "Gb/sec"
            if unit == "byte":
                val = val / (1024 **2)
                unit = "Mb"
            final_dict[f'{name} ({unit})'].append(val)
            
    final_df = pd.DataFrame(final_dict,index=index_list)
    out_file = Path(args.output_file)
    if not out_file.parent.exists:
        Path.mkdir(out_file.parent,exist_ok=True)
    final_df.to_csv(out_file)
    
    
        
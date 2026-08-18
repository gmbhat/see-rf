import pickle
from ReadFile import LoadData
import argparse , csv
from Test_Board import Test

def parse_args():
    # parser.add_argument("--dataset_name", type=str, default= "Epilepsy", help = "The Dataset name")
    # parser.add_argument("--num_exits", type=int, help = "The number of exits")
    # parser.add_argument("--proportions", type=float, help = "Data proportions", nargs="+")
    # parser.add_argument("--th_combination", type=float, help = "Threshold combination", nargs="+")
  
    parser = argparse.ArgumentParser(description = "RF-H Inference")
    parser.add_argument("--dataset_name", type=str, default= "Epilepsy", help = "The Dataset name")
    parser.add_argument("--n_est", type=int,default=50, help = "The number of estimators")
    parser.add_argument("--max_depth", type=int, default=100, help = "The max depth")
    parser.add_argument("--num_exits", type=int,  default= 3 ,help = "The number of exits")
    parser.add_argument("--tree_splits", type=list, default=[0.3, 0.7 , 0.9] ,help = "Tree splits")
    parser.add_argument("--proportions", type=list, default=[0.25, 0.5, 0.75] ,help = "Data proportions",  nargs="+")
    parser.add_argument("--th_combination", type=list, default=[0.34657359027997264, 0.6931471805599453], help = "Threshold combination", nargs="+")
 
    return parser.parse_args()

def write_content_to_file(file, content, header): # the content is a list of dictionaries
    writer = csv.writer(file)
    for line in content:
        row = [line[key] for key in header]
        writer.writerow(row)


def add_header(file, header):
    writer = csv.writer(file)
    writer.writerow(header)
    
    
if __name__ =="__main__":
    
    args = parse_args()
    classData = LoadData()
    classData.Read(args.dataset_name)
    classData.SplitData()

    
    with open(f"C:\\Users\\negar.haghpanahi\\OneDrive - Washington State University (email.wsu.edu)\\WSU\\Summer-2025\\RandomForest-Early-Exit\\Code\\SequentialRandomForest-main\\PKL_Saved_Files\\{args.dataset_name}_trained_model.pkl", "rb") as f:
        all_models = pickle.load(f,encoding='latin1')
        
    with open(f"C:\\Users\\negar.haghpanahi\\OneDrive - Washington State University (email.wsu.edu)\\WSU\\Summer-2025\\RandomForest-Early-Exit\\Code\\SequentialRandomForest-main\\PKL_Saved_Files\\{args.dataset_name}_trained_results.pkl", "rb") as f:
        current_config_training_metrics = pickle.load(f)
        
    all_result = Test(classData.GetTestX() , classData.GetYtest() ,all_models ,  args )
    
 
    output_file = f'{args.dataset_name}_accuracy_results.csv'
    # header = ['t_start','t1', 't2', 't3', 't4', 'total', 'true_label', 'prediction', 'correctness', 'exit_taken', 'data%']
        # Base header keys
    header = [
        't_start',
    ]

    # Dynamically add 't1', 't2', ... based on the number of exits + 1 (for total stages)
    # If num_exits = 3, this adds 't1', 't2', 't3'
    for i in range(1, args.num_exits + 1):
        header.append(f't{i}')

    # Add remaining keys
    header.extend([
        'total', 'true_label', 'prediction', 'correctness', 
        'exit_level', 'window_num', 'data%'
    ])
    with open(output_file, "w", newline="") as f1:
        add_header(f1, header)
        write_content_to_file(f1, all_result, header)
                
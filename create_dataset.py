import argparse
from dataset.datasets import DATASETS
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        required=True)
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='skip downloading base vectors')
    parser.add_argument(
        '--download',
        action='store_true',
        help='download dataset')
    parser.add_argument(
        '--para',
        action='store_true',
        help='creating parameter json describe dataset')
    args = parser.parse_args()
    
    
    
    if args.download:
        ds = DATASETS[args.dataset]()
        ds.prepare(True if args.skip_data else False)
    if args.para:
        ds = DATASETS[args.dataset]()
        para={
        "dataset":args.dataset,
        "nb":ds.nb,     
        "nq":ds.nq,
        "d":ds.d,
        "dtype":ds.dtype,
        "ds_fn":ds.ds_fn,
        "qs_fn":ds.qs_fn,
        "gt_fn":ds.gt_fn,
        "basedir":ds.basedir
        }
        filename = args.dataset + ".json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(para, f, ensure_ascii=False, indent=4)
        
        

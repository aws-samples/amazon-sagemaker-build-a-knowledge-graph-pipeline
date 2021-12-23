import os
import sys
import logging
import glob
import pathlib
import tarfile
import json
import argparse


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

device = None


def list_arg(raw_value):
    """argparse type for a list of strings"""
    return str(raw_value).split(",")


def parse_args():
    resconfig = {}
    try:
        with open("/opt/ml/config/resourceconfig.json", "r") as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        logger.error("/opt/ml/config/resourceconfig.json not found.  current_host is unknown.")
        pass  # Ignore

    # Local testing with CLI args
    parser = argparse.ArgumentParser(description="Process")

    parser.add_argument(
        "--hosts",
        type=list_arg,
        default=resconfig.get("hosts", ["unknown"]),
        help="Comma-separated list of host names running the job",
    )
    parser.add_argument(
        "--current-host",
        type=str,
        default=resconfig.get("current_host", "unknown"),
        help="Name of this host running the job",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="/opt/ml/processing/input/data",
    )
    parser.add_argument(
        "--input-model",
        type=str,
        default="/opt/ml/processing/input/model",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="/opt/ml/processing/output",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
    )
    
    parser.add_argument(
        "--bert-model-name",
        type=str,
        default="bert-base-chinese"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256
    )
    
    parser.add_argument(
        "--bert-dict-len",
        type=int,
        default=21127
    )
    
    parser.add_argument(
        "--word-emb", 
        type=int, 
        default=128
    )

    return parser.parse_args()

###################################
### SAGEMKAER LOAD MODEL FUNCTION
###################################

def model_fn(args):
    subject_path = os.path.join(args.input_model, 'subject.pt')
    object_path = os.path.join(args.input_model, 'object.pt')
    id2predicate_path = os.path.join(args.input_model, 'resources', 'id2predicate.json')
    if os.path.isfile(subject_path):
        logger.info(f"Model locates at: {args.input_model}")
    else:
        logger.error(f"Can't find model at {args.input_model}")
        logger.error(f"files under {args.input_model}: {os.listdir(args.input_model)}")
        logger.error(f"files in up folder: {os.listdir(os.path.join(args.input_model, '..'))}")
    if os.path.isfile(id2predicate_path):
        logger.info(f"id2predicate locates at: {id2predicate_path}")
        id2predicate = json.load(open(id2predicate_path))
    else:
        logger.error(f"Can't find id2predicates at {id2predicate_path}")
        id2predicate = json.load(open('./model/resources/id2predicate.jso'))
    subject_model = SubjectModel(args.bert_dict_len, args.word_emb).to(device)
    object_model = ObjectModel(args.word_emb, len(id2predicate)).to(device)
    subject_model.load_state_dict(torch.load(subject_path, map_location=device))
    object_model.load_state_dict(torch.load(object_path, map_location=device))
    return (subject_model, object_model, id2predicate)


def process(args):
    logger.info("Current host: {}".format(args.current_host))

    logger.info("input_data: {}".format(args.input_data))
    logger.info("input_model: {}".format(args.input_model))

    logger.info("Listing contents of input model dir: {}".format(args.input_model))
    input_files = os.listdir(args.input_model)
    logger.info(input_files)
    model_tar_path = "{}/model.tar.gz".format(args.input_model)
    model_tar = tarfile.open(model_tar_path)
    model_tar.extractall(args.input_model)
    model_tar.close()
    
    subject_model, object_model, id2predicate = model_fn(args)
    
    with open(os.path.join(args.input_data, 'test.json')) as f:
        test_data = json.load(f)
    dataset = DevDataset(test_data, args.bert_model_name, args.max_seq_length)
    loader = DataLoader(
        dataset=dataset,  
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=1,
        collate_fn=dev_collate_fn,
        multiprocessing_context='spawn',
    )
    
    f1, precision, recall = evaluate(subject_model, object_model, loader, id2predicate, epoch=0, writer=None, device=device)
    eval_metrics = {
       'f1': f1,
        'precision': precision,
        'recall': recall
    }
    metrics_path = os.path.join(args.output_data, 'metrics')
    pathlib.Path(metrics_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Dumping evaluation results to {metrics_path}/evaluation.json")
    with open(f"{metrics_path}/evaluation.json", 'w') as f:
        json.dump(eval_metrics, f, indent=4, ensure_ascii=False)
    if os.path.isfile(f"{metrics_path}/evaluation.json"):
        logger.info("Dumping succeed")
    else:
        logger.error(f"Failed to save evaluation results at {metrics_path}/evaluation.json")


if __name__ == "__main__":
    
    args = parse_args()
    logger.info("Loaded arguments:")
    print(args)
    logger.info("Environment variables:\n", os.environ)
    
    if args.source_dir:
        logger.info("Listing contents of input source dir: {}".format(args.source_dir))
        source_tar_path = f"{args.source_dir}/sourcedir.tar.gz"
        if os.path.isfile(source_tar_path):
            logger.info(f"Extract {source_tar_path} to {args.source_dir}")
            source_tar = tarfile.open(source_tar_path)
            source_tar.extractall(args.source_dir)
            source_tar.close()
        sys.path.insert(1, args.source_dir)
    else:
        logger.error("source dir is None")

    os.system(f"pip install -r {args.source_dir}/requirements.txt")
    from utils import evaluate
    from dataset import DevDataset, dev_collate_fn
    from model import SubjectModel, ObjectModel
    
    import torch
    from torch.utils.data import DataLoader
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    process(args)
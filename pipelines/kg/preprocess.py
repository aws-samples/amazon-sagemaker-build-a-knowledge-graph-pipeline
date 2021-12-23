"""Feature engineers the DuIE dataset."""
import argparse
import logging
import os
import pathlib
import json

import boto3
from tqdm import tqdm
from sklearn.model_selection import train_test_split


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()
logger.info(f"Current file path: {dir_path}")
logger.info(f"files under current file path:")
for file in os.listdir(dir_path):
    logger.info(file)
logger.info(f"Working directory: {cwd}")
logger.info(f"files under working directory:")
for file in os.listdir(cwd):
    logger.info(file)


# helper functions to upload data to s3
def write_to_s3(filename, bucket, prefix):
    import boto3
    # put one file in a separate folder. This is helpful if you read and prepare data with Athena
    filename_key = filename.split("/")[-1]
    key = os.path.join(prefix, filename_key)
    s3 = boto3.resource('s3')
    return s3.Bucket(bucket).upload_file(filename, key)


def upload_to_s3(bucket, prefix, filename):
    url = "s3://{}/{}".format(bucket, os.path.join(prefix, filename.split('/')[-1]))
    print("Writing to {}".format(url))
    write_to_s3(filename, bucket, prefix)


def trans(raw, processed, val_split):
    # Read raw schema
    schemas = set()
    with open(f"{raw}/schema.json") as f:
        for l in f:
            a = json.loads(l)
            schemas.add(a['predicate'])
    
    # Create id:predicate dicts
    id2predicate = {i+1:j for i,j in enumerate(schemas)}
    id2predicate[0] = 'UNK'
    predicate2id = {j:i for i,j in id2predicate.items()}

    # Save processed schema
    with open(f"{processed}/schema.json", 'w', encoding='utf-8') as f:
        json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)

    logger.info("Processing raw train data...")
    raw_data = []
    with open(f"{raw}/train.json") as f:
        for l in tqdm(f):
            a = json.loads(l)
            raw_data.append(
                {
                    'text': a['text'],
                    'spo_list': [(i['subject'], i['predicate'], i['object']['@value']) for i in a['spo_list']]
                }
            )
    # Split train and dev
    train_data, val_data = train_test_split(raw_data, test_size=val_split)
    logger.info(f"Dumping processed train data into {processed}/train.json")
    with open(f"{processed}/train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    with open(f"{processed}/val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4, ensure_ascii=False)
    logger.info("Processing raw test data...")
    test_data = []
    with open(f"{raw}/dev.json") as f: # test data are in dev.json
        for l in tqdm(f):
            a = json.loads(l)
            test_data.append(
                {
                    'text': a['text'],
                    'spo_list': [(i['subject'], i['predicate'], i['object']['@value']) for i in a['spo_list']]
                }
            )
    logger.info(f"Dumping processed test data into {processed}/test.json")
    with open(f"{processed}/test.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
        
    logger.info(f"train data:\t {len(train_data)} pieces")
    logger.info(f"val data:\t {len(val_data)} pieces")
    logger.info(f"test data:\t {len(test_data)} pieces")


if __name__ == "__main__":
    
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=False)
    parser.add_argument("--validation-split", type=float, default=0.05)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing/ie"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    if input_data.startswith('s3://'):
        bucket = input_data.split("/")[2]
        key = "/".join(input_data.split("/")[3:])
        logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
        fn = f"{base_dir}/data/duie.zip"
        s3 = boto3.resource("s3")
        s3.Bucket(bucket).download_file(key, fn)
    else:
        # it should be a local file
        if os.path.isfile(args.input_data):
            logger.info(f"Raw data locates at {args.input_data}")
            fn = args.input_data
        elif os.path.isdir(args.input_data):
            logger.error(f"{args.input_data} should be the file location.\n trying to load {args.input_data}/DuIE_2_0.zip")
            logger.info(f"files under {args.input_data}: {os.listdir(args.input_data)}")
            fn =  f"{args.input_data}/DuIE_2_0.zip"
        else:
            logger.error(f"{args.input_data} doesn't exist")
            logger.info(f"files in parent folder {args.input_data+'/..'}: {os.listdir(args.input_data+'/..')}")
            fn = args.input_data

    logger.info("Unzipping dowloaded data...")
    raw = f"{base_dir}/data/raw"
    pathlib.Path(raw).mkdir(parents=True, exist_ok=True)
    os.system(f"unzip -j {fn} -d {raw}")
    logger.info(f"Data unzipped to {raw}")
    
    processed = f"{base_dir}/data/processed"
    pathlib.Path(processed).mkdir(parents=True, exist_ok=True)
    
    # Transform raw data
    trans(raw, processed, args.validation_split)
    # Delete downloaded and raw data
    os.unlink(fn)
    os.unlink(f"{raw}/schema.json")
    os.unlink(f"{raw}/train.json")
    os.unlink(f"{raw}/dev.json")
    
    # Upload processed data to s3
    if args.output_dir is not None:
        bucket = args.output_dir.split("/")[2]
        prefix = "/".join(args.output_dir.split("/")[3:])
        upload_to_s3(bucket, prefix, f"{processed}/train.json")
        upload_to_s3(bucket, prefix, f"{processed}/val.json")
        upload_to_s3(bucket, prefix, f"{processed}/test.json")
        upload_to_s3(bucket, prefix, f"{processed}/schema.json")
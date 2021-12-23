'''
Programatically ingest data into Neptune graph database
1. Locate transformed data and schema
2. Convert transformed data into nodes and edges
3. Upload nodes and edges csv files to S3
4. Use loader tool provided by Neptune to ingest data
'''
import os
import json
import argparse
import logging
import pathlib
import subprocess
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformed-data', type=str, help='local path to batch transformed data')
    parser.add_argument('--raw-dataset', type=str, help='local path to raw dataset, this is to load schema')
    parser.add_argument('--bucket', required=False, type=str, help='s3 bucket to save generated files')
    parser.add_argument('--save-prefix', required=False, type=str, help='s3 prefix folder to save generated files')
    parser.add_argument('--loadfroms3-role', type=str, required=False, help='an iam role attached to Neptune to access aws services')
    parser.add_argument('--neptune-endpoint',required=False, type=str, help='neptune cluster endpoint')
    parser.add_argument('--neptune-port',required=False, type=str, default='8182', help='neptune port')
    parser.add_argument('--neptune-region',required=False, type=str, default='us-east-2', help='region of Neptune database') 
    return parser.parse_known_args()

# currently id is constructed naively.
def node_name2id(entity_type, entity_name):
    return 'node_' + entity_type + '_' + entity_name

# helper functions to upload data to s3
def write_to_s3(filename, bucket, prefix):
    import boto3
    # put one file in a separate folder. This is helpful if you read and prepare data with Athena
    filename_key = filename.split("/")[-1]
    key = os.path.join(prefix, filename_key)
    s3 = boto3.resource('s3')
    return s3.Bucket(bucket).upload_file(filename, key)

def upload_to_s3(bucket, prefix, filename):
    url = "s3://{}/".format(bucket, os.path.join(prefix, filename.split('/')[-1]))
    print("Writing to {}".format(url))
    write_to_s3(filename, bucket, prefix)

def convert_to_nodes_and_edges(triplets_path, schema_path):
    '''
    Convert json file of (subject, predicate and object) to graph nodes and edges
    '''
    rel_df = pd.read_json(triplets_path)
    rel_dict = {}
    with open(schema_path) as f:
        for l in f:
            rel = json.loads(l)
            #schemas.add(a['predicate'])
            predicate = rel['predicate']
            sub_type = rel['subject_type']
            obj_type = rel['object_type']['@value']
            rel_dict[predicate] = {'subject_type': sub_type, 'object_type': obj_type}

    node_df = pd.DataFrame({'~id':[], '~label':[], 'name': []})
    edge_df = pd.DataFrame({'~id':[], '~from':[], '~to':[], '~label':[]})

    node_dict = {}
    for idx, row in rel_df.iterrows():
        sub = row['subject']
        obj = row['object']
        rel = row['predicate']
        sub_type = rel_dict[rel]['subject_type']
        obj_type = rel_dict[rel]['object_type']
        sub_id = 'node_' + sub_type + '_' + sub
        obj_id = 'node_' + obj_type + '_' + obj
        # order matter: ~id, ~label, name
        node_dict[sub_id] = [sub_type, sub]
        node_dict[obj_id] = [obj_type, obj]
        edge_id = 'edge_' + rel + '_' + sub_id + '_' + obj_id
        edge_df.loc[len(edge_df)] = [edge_id, sub_id, obj_id, rel]
        
    for key, val in node_dict.items():
        node_df.loc[len(node_df)] = [key, val[0], val[1]]  

    logger.info("We have scanned {} nodes and {} relations".format(len(node_df), len(edge_df)))

    return node_df, edge_df


if __name__ == '__main__':
    args, _ = parse_args()
    if os.path.isfile(args.raw_dataset):
        logger.info(f"Raw data locates at {args.raw_dataset}")
        fn = args.raw_dataset
    elif os.path.isdir(args.raw_dataset):
        fn =  f"{args.raw_dataset}/DuIE_2_0.zip"
    else:
        logger.info(f"{args.raw_dataset} does not exist")
    
    logger.info("Unzipping dowloaded data...")
    base_dir = "/opt/ml/processing/ie"
    base_dir = "data/temp"
    raw = f"{base_dir}/data/raw"
    pathlib.Path(raw).mkdir(parents=True, exist_ok=True)
    os.system(f"unzip -j {fn} -d {raw}")
    logger.info(f"Data unzipped to {raw}")

    schema_path = f"{raw}/schema.json"
    
    if os.path.isfile(args.transformed_data):
        logger.info(f"Transformed data locates at {args.transformed_data}")
        transformed = args.transformed_data
    elif os.path.isdir(args.transformed_data):
        transformed =  os.path.join(args.transformed_data, os.listdir(args.transformed_data)[0])
        logger.info(f"Files under transformed {os.listdir(args.transformed_data)}")
        logger.info(f"Taking {transformed} as output from transform step")
    else:
        logger.info(f"{args.transformed_data} does not exist")
        transformed = None
    
    node_df, edge_df = convert_to_nodes_and_edges(transformed, schema_path)

    pathlib.Path(f"{base_dir}/data/graph").mkdir(parents=True, exist_ok=True)
    local_node_path = f"{base_dir}/data/graph/nodes.csv"
    local_edge_path = f"{base_dir}/data/graph/edge.csv"
    node_df.to_csv(local_node_path, index=False)
    edge_df.to_csv(local_edge_path, index=False)

    upload_to_s3(args.bucket, args.save_prefix, local_node_path)
    upload_to_s3(args.bucket, args.save_prefix, local_edge_path)

    load_script = f"""curl -X POST \
    -H 'Content-Type: application/json' \
    https://{args.neptune_endpoint}:8182/loader -d '
    {{
      "source" : "s3://{args.bucket}/{args.save_prefix}/",
      "format" : "csv",
      "iamRoleArn" : "{args.loadfroms3_role}",
      "region" : "{args.neptune_region}",
      "failOnError" : "FALSE",
      "parallelism" : "MEDIUM",
      "updateSingleCardinalityProperties" : "FALSE",
      "queueRequest" : "TRUE",
      "dependencies" : []
    }}'
    """

    logger.info('Running script (this must be run in the same VPC):')
    logger.info(load_script)

    subprocess.run(load_script, shell=True)


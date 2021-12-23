'''
Create or access a Neptune Graph database.

Permissions Required:
iam:GetAccountSummary on resource: *
iam:ListAccountAliases on resource: *
iam:PassRole on resource: * with iam:PassedToService restricted to rds.amazonaws.com
iam:CreateRole
rds:DescribeDBClusters
rds:CreateDBClusters
'''
import os
import time
import argparse
import json
import logging
import pathlib
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-cluster-identifier', type=str, default='kg-neptune', help='Neptune cluster name. If not exists, one with this name will be created.')
    parser.add_argument('--db-instance-suffix', type=str, default='instance-1', help='If there does not exist any instance within the cluster, an instance with id [db-cluster-identifier]-[db-instance-suffix] will be created.')
    parser.add_argument('--db-instance-class', type=str, default='db.t3.medium')
    parser.add_argument('--load-from-s3-role-name', type=str, default='NeptuneLoadFromS3', help='The name of role that allows Neptune to access S3. A role with this name will be created if it does not exist.')
    parser.add_argument('--output-neptune-metadata-dir', type=str, default='/opt/ml/processing/output/',)
    parser.add_argument('--output-neptune-metadata-filename', type=str, default='neptune_meta.json')
    return parser.parse_known_args()


def get_or_create_db_cluster(db_cluster_identifier):
    neptune = boto3.client('neptune')
    try:
        response = neptune.describe_db_clusters(DBClusterIdentifier=db_cluster_identifier)
        db_cluster = response['DBClusters'][0]
    except ClientError as e:
        if e.response["Error"]["Code"] != 'DBClusterNotFoundFault':
            raise e
        print(f"Neptune Cluster {db_cluster_identifier} does not exist.")
        print(f"Trying to create a Neptune Cluster with identifier {db_cluster_identifier}")
        response = neptune.create_db_cluster(
            DBClusterIdentifier=db_cluster_identifier, 
            Engine='neptune'
        )
        db_cluster = response['DBCluster']
    return db_cluster


def get_or_create_db_instance(db_cluster_identifier, db_instance_suffix, db_instance_class):
    neptune = boto3.client('neptune')
    db_instance_identifier = f"{db_cluster_identifier}-{db_instance_suffix}"
    try:
        response = neptune.describe_db_instances(DBInstanceIdentifier=db_instance_identifier)
        db_instance = response['DBInstances'][0]
    except ClientError as e:
        if e.response["Error"]["Code"] != 'DBInstanceNotFound':
            raise e
        print(f"Trying to create a Neptune DB instance with identifier {db_instance_identifier}")
        response = neptune.create_db_instance(
            DBInstanceIdentifier=db_instance_identifier,
            DBInstanceClass=db_instance_class,
            Engine='neptune',
            DBClusterIdentifier=db_cluster_identifier,
        )
        db_instance = response['DBInstance']
    return db_instance


def get_or_create_loadfroms3_role(load_from_s3_role_name):
    iam = boto3.client("iam")
    s3_read_only_policy_arn = 'arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess'

    assume_role_policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "",
                "Effect": "Allow", 
                "Principal": {
                    "Service": [
                      "rds.amazonaws.com"
                    ]
                  },
                "Action": "sts:AssumeRole"
            }
        ],
    }

    try:
        iam_role_loadfroms3 = iam.create_role(
            RoleName=load_from_s3_role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_doc),
            Description="Allow Amazon Neptune to Access Amazon S3 Resources",
        )
        # attach s3 read only policy
        response = iam.attach_role_policy(
            RoleName=load_from_s3_role_name,
            PolicyArn=s3_read_only_policy_arn
        )
        print('Role:\n', iam_role_loadfroms3)
        print('Attach Policy Response:\n', response)
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            print("Role already exists")
            iam_role_loadfroms3 = iam.get_role(
                RoleName=load_from_s3_role_name
            )
            print(iam_role_loadfroms3)
        else:
            print("Unexpected error: %s" % e)
    return iam_role_loadfroms3


def create_s3_endpoint_if_not_exist(db_cluster_region, vpc_id):
    ec2 = boto3.client('ec2')
    s3_service_name = f"com.amazonaws.{db_cluster_region}.s3"
    # Check endpoints existence
    response = ec2.describe_vpc_endpoints(
        Filters=[
            {
                'Name': 'service-name',
                'Values': [s3_service_name]
            },
            {
                'Name': 'vpc-id',
                'Values': [vpc_id]
            },
            {
                'Name': 'vpc-endpoint-type',
                'Values': ['Gateway']
            }
        ]
    )
    if len(response['VpcEndpoints']) > 0:
        vpc_endpoint = response['VpcEndpoints'][0]
    else:
        print('Trying to create an VPC endpoint:')
        response = ec2.create_vpc_endpoint(
            VpcEndpointType='Gateway',
            VpcId=vpc_id,
            ServiceName=s3_service_name,
        )
        vpc_endpoint = response['VpcEndpoint']
    return vpc_endpoint


def get_instances_by_cluster(db_cluster_identifier):
    neptune = boto3.client('neptune')
    response = neptune.describe_db_instances(
        Filters=[
            {
                'Name': 'db-cluster-id',
                'Values': [db_cluster_identifier]
            },
            {
                'Name': 'engine',
                'Values': ['neptune']
            }
        ]
    )
    db_instances = response['DBInstances']
    return db_instances


def add_role(db_cluster_identifier, iam_role_loadfroms3_arn):
    neptune = boto3.client('neptune')
    try:
        response = neptune.describe_db_clusters(DBClusterIdentifier=db_cluster_identifier)
        associated_roles = response['DBClusters'][0]['AssociatedRoles']
        add_role =True
        if len(associated_roles) > 0:
            for role in associated_roles:
                if role['RoleArn'].find(iam_role_loadfroms3_arn) >= 0:
                    add_role = False
                    break;
                    
        if add_role:
            response = neptune.add_role_to_db_cluster(
                DBClusterIdentifier=db_cluster_identifier,
                RoleArn=iam_role_loadfroms3_arn
            )
            
    except ClientError as e:
        raise e


if __name__ == '__main__':
    args, _ = parse_args()
    
    db_cluster = get_or_create_db_cluster(args.db_cluster_identifier)
    # Wait for Neptune cluster to come online
    while db_cluster['Status'] == 'creating':
        logger.info(f"Cluster {args.db_cluster_identifier} is in status \'creating\', waiting...")
        time.sleep(30) # check status every 30 seconds
        db_cluster = get_or_create_db_cluster(args.db_cluster_identifier)
    logger.info(f"Cluster {args.db_cluster_identifier} is now in status \'{db_cluster['Status']}\'")
    
    db_instances = get_instances_by_cluster(args.db_cluster_identifier)
    if len(db_instances) > 0:
        logger.info(f"There exists instances within cluster {args.db_cluster_identifier}")
        db_instance = db_instances[0]
    else:
        db_instance = get_or_create_db_instance(args.db_cluster_identifier, args.db_instance_suffix, args.db_instance_class)
    
    # Wait for neptune instance to come online
    while db_instance['DBInstanceStatus'] == 'creating':
        logger.info(f"Instance {db_instance['DBInstanceIdentifier']} is in status \'creating\', waiting...")
        time.sleep(15) # check status every 15 seconds
        db_instance = get_or_create_db_instance(args.db_cluster_identifier, args.db_instance_suffix, args.db_instance_class)
    
    logger.info(f"Instance {db_instance['DBInstanceIdentifier']} is now in status \'{db_instance['DBInstanceStatus']}\'")
    
    iam_role_loadfroms3 = get_or_create_loadfroms3_role(args.load_from_s3_role_name)
    time.sleep(10)
    iam_role_loadfroms3_arn = iam_role_loadfroms3['Role']['Arn']
    add_role(args.db_cluster_identifier, iam_role_loadfroms3_arn)
    
    roleActive = False
    while not roleActive:
        roleActive = True
        db_cluster_info = get_or_create_db_cluster(args.db_cluster_identifier)
        ass_roles = db_cluster_info['AssociatedRoles']
        if len(ass_roles) > 0:
            for ass_role in ass_roles:
                if ass_role['Status'] != 'ACTIVE':
                    print(f'{ass_role} is not active')
                    roleActive = False
                    time.sleep(15)
                    break
    
    
    db_cluster_arn = db_cluster['DBClusterArn']
    db_cluster_region = db_cluster_arn.split(':')[3]
    vpc_id = db_instance['DBSubnetGroup']['VpcId']
    vpc_sg = db_instance['VpcSecurityGroups'][0]['VpcSecurityGroupId']
    vpc_subnets = db_instance['DBSubnetGroup']['Subnets'][0]['SubnetIdentifier']
    
    s3_vpc_endpoint = create_s3_endpoint_if_not_exist(db_cluster_region, vpc_id)
    s3_vpc_endpoint_id = s3_vpc_endpoint['VpcEndpointId']
    
    db_cluster_endpoint = db_cluster['Endpoint']
    db_cluster_port = db_cluster['Port']
    
    neptune_metadata = {
        'cluster_arn': db_cluster_arn,
        'cluster_region': db_cluster_region,
        'cluster_endpoint': db_cluster_endpoint,
        'cluster_port': str(db_cluster_port),
        'vpc_id': vpc_id,
        'vpc_subnets': vpc_subnets,
        'vpc_sg': vpc_sg,
        's3_endpoint_id': s3_vpc_endpoint_id,
        'role_loadfroms3_arn': iam_role_loadfroms3_arn
    }
    
    pathlib.Path(args.output_neptune_metadata_dir).mkdir(parents=True, exist_ok=True)
    neptune_metadata_path = os.path.join(args.output_neptune_metadata_dir, args.output_neptune_metadata_filename)
    logger.info(f"Dumping neptune metadata to {neptune_metadata_path}")
    with open(neptune_metadata_path, 'w') as f:
        json.dump(neptune_metadata, f, indent=4, ensure_ascii=False)
        
    logger.info("Metadata file dumped.")
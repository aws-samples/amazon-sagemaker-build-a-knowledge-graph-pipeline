
"""An workflow pipeline script for Knowledge Graph generation pipeline.

                                                   . RegisterModel
                                               . -
                                              .    . CreateModel -> BatchTransform -> NeptuneBulkload
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . - AlertDevTeam

Implements a get_pipeline(**kwargs) method.
"""
import subprocess
import sys

import os
import time

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import (
    TrainingInput,
    CreateModelInput,
    TransformInput
)

from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep,
    TransformStep
)
from sagemaker.workflow.steps import CacheConfig
from sagemaker.debugger import Rule, ProfilerRule, rule_configs
from sagemaker.debugger import DebuggerHookConfig
from sagemaker.debugger import ProfilerConfig, FrameworkProfile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.pytorch import PyTorchModel
from sagemaker.model import FrameworkModel
from sagemaker.transformer import Transformer

import pandas as pd

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
print(f'BASE_DIR: {BASE_DIR}')

def get_sagemaker_client(region):
    """
     Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client

def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_step_processing(bucket, region, role, params):
    '''
    params: 
        raw_input_dataset
        output_dir
        processing_instance_count
        processing_instance_type
    '''
    raw_input_dataset = params['raw_input_dataset']
    output_dir = params['output_dir']
    processing_instance_count = params['processing_instance_count']
    processing_instance_type = params['processing_instance_type']
    processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        env={"AWS_DEFAULT_REGION": region},
        )
    processing_inputs = [
        ProcessingInput(
            input_name="raw",
            source=raw_input_dataset,
            destination="/opt/ml/processing/ie/data/raw",
            s3_data_distribution_type="ShardedByS3Key",
            )
        ]
    processing_outputs = [
        ProcessingOutput(
            output_name="train",
            destination = output_dir,
            s3_upload_mode="EndOfJob",
            source="/opt/ml/processing/ie/data/processed",
        )
    ]
    processing_step = ProcessingStep(
        name="Processing",
        code=os.path.join(BASE_DIR, "preprocess.py"),
        processor=processor,
        inputs=processing_inputs,
        outputs=processing_outputs,
        job_arguments=[
            "--input-data",
            processing_inputs[0].destination, # /opt/ml/processing/ie/data/raw
        ],
    )
    return processing_step


def get_step_training(bucket, region, role, params, dependencies):
    '''
    params:
        train_instance_type
        train_instance_count
        epochs
        learning_rate
        batch_size
    dependencies: 
        'step_process': processing_step
    '''
    train_instance_type = params['train_instance_type']
    train_instance_count = params['train_instance_count']
    epochs = params['epochs']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    # Setup Metrics To Track Model Performance
    metric_definitions = [
        {'Name': 'eval:f1', 'Regex': 'f1: ([0-9\\.]+)'},
        {'Name': 'eval:prec', 'Regex': 'precision: ([0-9\\.]+)'},
        {'Name': 'eval:recall', 'Regex': 'recall: ([0-9\\./]+)'}
    ]
    # Setup Debugger and Profiler
    # Define Debugger Rules as described here: https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html
    debugger_hook_config = DebuggerHookConfig(
        s3_output_path="s3://{}/ie-baseline/debug".format(bucket),
    )
    profiler_config = ProfilerConfig(
        system_monitor_interval_millis=500,
        framework_profile_params=FrameworkProfile(local_path="/opt/ml/output/profiler/", start_step=5, num_steps=10),
    )
    rules = [ProfilerRule.sagemaker(rule_configs.ProfilerReport())]
    # Define a Training Step to Train a Model
    estimator = PyTorch(
        entry_point='train.py', # relative path to unzipped sourcedir, don't specify as a local path
        source_dir=BASE_DIR,
        role=role,
        instance_type=train_instance_type, # ml.c5.4xlarge, ml.g4dn.4xlarge
        instance_count=train_instance_count,
        framework_version='1.8.1',
        py_version='py3',
        output_path=f"s3://{bucket}/ie-baseline/outputs",
        code_location=f"s3://{bucket}/ie-baseline/source/train", # where custom code will be uploaded 
        hyperparameters={
            'epochs': epochs,
            'use-cuda': True,
            'batch-size': batch_size,
            'learning-rate': learning_rate
        },
        metric_definitions = metric_definitions,
        debugger_hook_config=debugger_hook_config,
        profiler_config=profiler_config,
        rules=rules
    )
    # Setup Pipeline Step Caching
    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")

    training_step = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=dependencies['step_process'].properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="application/json",
            ),
        },
        cache_config=cache_config,
    )
    return training_step


def get_step_evaluation(bucket, region, role, params, dependencies, properties):
    '''
    params:
        evaluation_instance_count
        evaluation_instance_type
    dependencies: 
        'step_train'
        'step_process'
    properties:
        evaluation_report
    '''
    evaluation_instance_count = params['evaluation_instance_count']
    evaluation_instance_type = params['evaluation_instance_type']
    evaluation_report = properties['evaluation_report']
    evaluation_processor = SKLearnProcessor(
        role=role,
        framework_version="0.23-1",
        instance_type=evaluation_instance_type,
        instance_count=evaluation_instance_count,
        env={"AWS_DEFAULT_REGION": region},
        max_runtime_in_seconds=7200,
    )
    evaluation_step = ProcessingStep(
        name="EvaluateModel",
        processor=evaluation_processor,
        code=os.path.join(BASE_DIR, "evaluate.py"),
        inputs=[
            ProcessingInput(
                input_name='model',
                source=dependencies['step_train'].properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model",
            ),
            ProcessingInput(
                input_name='data',
                source=dependencies['step_process'].properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/data",
            ),
            ProcessingInput(
                input_name='source',
                source=dependencies['step_train'].arguments['HyperParameters']['sagemaker_submit_directory'][1:-1],
                destination="/opt/ml/processing/input/source/train"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="metrics", s3_upload_mode="EndOfJob", source="/opt/ml/processing/output/metrics/"
            ),
        ],
        job_arguments=[
            "--max-seq-length",
            "128",
            "--source-dir",
            "/opt/ml/processing/input/source/train"
        ],
        property_files=[evaluation_report]
    )
    return evaluation_step


def get_step_create_model(bucket, region, role, sess, params, dependencies):
    '''
    params:
        transform_model_name
        inference_instance_type
    dependencies: 'step_train'
    '''
    transform_model_name = params['transform_model_name']
    inference_instance_type = params['inference_instance_type']

    model = PyTorchModel(
        name=transform_model_name,
        model_data=dependencies['step_train'].properties.ModelArtifacts.S3ModelArtifacts,
        framework_version='1.3.1',
        py_version='py3',
        role=role,
        entry_point='inference.py',
        source_dir=BASE_DIR,
        sagemaker_session=sess
    )
    create_inputs = CreateModelInput(
        instance_type=inference_instance_type,
        accelerator_type="ml.eia1.medium",
    )
    step_create_model = CreateModelStep(
        name="CreateKgGenModel",
        model=model,
        inputs=create_inputs,
    )
    return step_create_model


def get_step_transform(bucket, region, role, params, dependencies):
    '''
    params:
        transform_instance_type
        batch_data
        transform_output_prefix
    dependencies: 'step_create_model'
    '''
    transform_instance_type = params['transform_instance_type']
    batch_data = params['batch_data']
    transform_output_prefix = params['transform_output_prefix']
    transformer = Transformer(
        model_name=dependencies['step_create_model'].properties.ModelName,
        instance_type=transform_instance_type,
        instance_count=1,
        output_path=f"s3://{bucket}/{transform_output_prefix}",
    )
    step_transform = TransformStep(
        name="KgTransform", transformer=transformer, inputs=TransformInput(data=batch_data)
    )
    return step_transform


def get_step_register_model(model_package_group_name, params, dependencies):
    '''
    params:
        model_approval_status
        transform_instance_type
        deploy_instance_type
    dependencies: 
        'step_evaluate'
        'step_train'
    '''
    model_approval_status = params['model_approval_status']
    transform_instance_type = params['transform_instance_type']
    deploy_instance_type = params['deploy_instance_type']
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                dependencies['step_evaluate'].arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )
    step_register = RegisterModel(
        name="KgRegisterModel",
        estimator=dependencies['step_train'].estimator,
        model_data=dependencies['step_train'].properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[deploy_instance_type],
        transform_instances=[transform_instance_type],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    return step_register


def get_step_create_db(bucket, region, role, params, dependencies, properties):
    '''
    params:
        db_cluster_identifier
        iam_loadfroms3_role_name
    dependencies:
        step_transform
    properties:
        neptune_metadata
    '''
    db_cluster_identifier = params['db_cluster_identifier']
    iam_loadfroms3_role_name = params['iam_loadfroms3_role_name']
    neptune_metadata = properties['neptune_metadata']

    processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        env={"AWS_DEFAULT_REGION": region},
    )
    
    output_neptune_metadata_dir = "/opt/ml/processing/output/"
    output_name = neptune_metadata.output_name
    output_neptune_metadata_filename = neptune_metadata.path.split('/')[-1]
    default_db_instance_suffix = 'instance-1'
    default_db_instance_class = 'db.t3.medium'
    create_db_step = ProcessingStep(
        name="RetrieveOrCreateNeptuneDB",
        code=os.path.join(BASE_DIR, 'createdb.py'),
        processor=processor,
        outputs=[
            ProcessingOutput(
                output_name=output_name, s3_upload_mode="EndOfJob", source=output_neptune_metadata_dir
            ),
        ],
        job_arguments=[
            "--db-cluster-identifier",
            db_cluster_identifier,
            "--db-instance-suffix",
            default_db_instance_suffix,
            "--db-instance-class",
            default_db_instance_class,
            "--load-from-s3-role-name",
            iam_loadfroms3_role_name,
            "--output-neptune-metadata-dir",
            output_neptune_metadata_dir,
            "--output-neptune-metadata-filename",
            output_neptune_metadata_filename
        ],
        property_files=[neptune_metadata]
    )
    create_db_step.add_depends_on([dependencies['step_transform']])
    return create_db_step


def get_step_bulkload(bucket, region, role, params, dependencies, properties):
    '''
    params:
        bulkload_instance_type
        raw_input_dataset
    dependencies:
        step_transform
        step_create_db
    properties:
        neptune_metadata
    '''
    bulkload_instance_type = params['bulkload_instance_type']
    raw_input_dataset = params['raw_input_dataset']
    savePrefix = "ie-baseline/graphData"
    neptune_metadata = properties['neptune_metadata']
    
    db_cluster_endpoint = JsonGet(
        step_name=dependencies['step_create_db'].name,
        property_file=neptune_metadata,
        json_path="cluster_endpoint",
    )
    db_cluster_port = JsonGet(
        step_name=dependencies['step_create_db'].name,
        property_file=neptune_metadata,
        json_path="cluster_port",
    )
    db_cluster_region = JsonGet(
        step_name=dependencies['step_create_db'].name,
        property_file=neptune_metadata,
        json_path="cluster_region",
    )
    iam_role_loadfroms3_arn = JsonGet(
        step_name=dependencies['step_create_db'].name,
        property_file=neptune_metadata,
        json_path="role_loadfroms3_arn",
    )
    db_cluster_sg = JsonGet(
        step_name=dependencies['step_create_db'].name,
        property_file=neptune_metadata,
        json_path="vpc_sg",
    )
    db_cluster_subnets = JsonGet(
        step_name=dependencies['step_create_db'].name,
        property_file=neptune_metadata,
        json_path="vpc_subnets",
    )
    
    network_configuration = sagemaker.network.NetworkConfig(enable_network_isolation=False, 
                                                        security_group_ids=[db_cluster_sg], 
                                                        subnets=[db_cluster_subnets], 
                                                        encrypt_inter_container_traffic=False)
    
    processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=bulkload_instance_type,
        instance_count=1,
        env={"AWS_DEFAULT_REGION": region},
        network_config=network_configuration
    )

    bulkload_inputs = [
        ProcessingInput(
            input_name="TransformedData",
            source=dependencies['step_transform'].transformer.output_path,
            destination="/opt/ml/processing/ie/data/transformed",
            s3_data_distribution_type="ShardedByS3Key",
        ),
        ProcessingInput(
            input_name="RawDataset",
            source=raw_input_dataset,
            destination="/opt/ml/processing/ie/data/raw/",
            s3_data_distribution_type="ShardedByS3Key",
        ),
    ]

    bulkload_step = ProcessingStep(
        name="NeptuneBulkload",
        code=os.path.join(BASE_DIR, 'bulkload.py'),
        processor=processor,
        inputs=bulkload_inputs,
        job_arguments=[
            "--transformed-data",
            bulkload_inputs[0].destination, # /opt/ml/processing/ie/data/raw
            "--raw-dataset",
            bulkload_inputs[1].destination, # DuIE_2_0.zip
            "--bucket",
            bucket,
            "--save-prefix",
            savePrefix,
            "--loadfroms3-role",
            iam_role_loadfroms3_arn,
            "--neptune-region",
            db_cluster_region,
            "--neptune-endpoint",
            db_cluster_endpoint,
            "--neptune-port",
            db_cluster_port
        ],
    )
    bulkload_step.add_depends_on([dependencies['step_create_db']])
    return bulkload_step


def get_step_alert(bucket, region, role, params, dependencies):
    '''
    params:
        alert_emails
        alert_phones
    dependencies:
        step_evaluate
    '''
    alert_topic = params['alert_topic']
    alert_msg = params['alert_msg']
    alert_emails = params['alert_emails']
    alert_phones = params['alert_phones']

    processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        env={"AWS_DEFAULT_REGION": region},
    )
    
    # TODO: get metrics from step_evaluate or step_evaluate
    alert_step = ProcessingStep(
        name="AlertDevTeam",
        code=os.path.join(BASE_DIR, 'alert.py'),
        processor=processor,
        job_arguments=[
            "--alert-topic",
            alert_topic,
            "--alert-message",
            alert_msg,
            "--alert-emails",
            alert_emails,
            "--alert_phones",
            alert_phones
        ],
    )
    return alert_step


def get_step_condition(params, dependencies, properties):
    '''
    params:
        min_f1_value
    dependencies: 
        'step_evaluate'
        'step_register'
        'step_create_model'
        'step_transform'
        'step_bulkload'
        'step_create_db'
        'step_alert'
    properties:
        evaluation_report
    '''
    min_f1_value = params['min_f1_value']
    evaluation_report = properties['evaluation_report']
    minimum_f1_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=dependencies['step_evaluate'].name,
            property_file=evaluation_report,
            json_path="f1",
        ),
        right=min_f1_value,  # accuracy
    )
    minimum_f1_condition_step = ConditionStep(
        name="F1Condition",
        conditions=[minimum_f1_condition],
        if_steps=[
            dependencies['step_register'], 
            dependencies['step_create_model'],
            dependencies['step_transform'], 
            dependencies['step_create_db'],
            dependencies['step_bulkload']
        ],  # success, continue with model registration
        else_steps=[
            dependencies['step_alert']
        ],  # fail, end the pipeline
    )
    return minimum_f1_condition_step


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket='sm-nlp-data',
    model_package_group_name="KgGenPackageGroup",
    pipeline_name="KnowledgeGraphGenerationPipeline",
    base_job_prefix="ie",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    print(f"SM role ARN: {sagemaker_project_arn}")
    pipeline_name = pipeline_name + str(int(time.time()))
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    sess = sagemaker_session
        
    # processing parameters
    raw_input_data_s3_uri = "s3://{}/ie-baseline/raw/DuIE_2_0.zip".format(default_bucket)
    processed_data_s3_uri = "s3://{}/ie-baseline/processed/".format(default_bucket)
    raw_input_dataset = ParameterString(name="InputDataset", default_value=raw_input_data_s3_uri)
    output_dir = ParameterString(name="ProcessingOutputData", default_value=processed_data_s3_uri,)
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.c5.2xlarge") #ml.c4.xlarge

    # train parameters
    train_instance_type = ParameterString(name="TrainInstanceType", default_value="ml.g4dn.4xlarge") 
    train_instance_count = ParameterInteger(name="TrainInstanceCount", default_value=1)
    epochs = ParameterString(name="Epochs", default_value='20')
    learning_rate = ParameterString(name="LearningRate", default_value='0.001')
    batch_size = ParameterString(name="BatchSize", default_value='64')

    # evaluate parameters
    evaluation_instance_count = ParameterInteger(name="EvaluationInstanceCount", default_value=1)
    evaluation_instance_type = ParameterString(name="EvaluationInstanceType", default_value="ml.c5.2xlarge")
    
    # alert parameters
    alert_topic = ParameterString(name="AlertTopic", default_value="KGPipelineAlert")
    alert_msg = ParameterString(name="AlertMsg", default_value="Didn't pass model evaluation.")

    # create model parameters
    transform_model_name = ParameterString(name="TransformModelName", default_value="transform-model-{}".format(int(time.time())))
    inference_instance_type = ParameterString(name="InferenceInstanceType", default_value="ml.c5.4xlarge") # ml.c5.4xlarge, ml.g4dn.16xlarge
 
    # batch transform parameters
    transform_instance_type = ParameterString(name="TransformInstanceType", default_value="ml.c5.4xlarge")
    batch_data = ParameterString(name="BatchData", default_value=f's3://{default_bucket}/psudo/psudo.json',)
    transform_output_prefix = ParameterString(name="TransformOutputPrefix", default_value='ie-baseline/outputs/transformed')
    
    # create db step
    db_cluster_identifier = ParameterString(name="NeptuneClusterIdentifier", default_value="kg-neptune")
    iam_loadfroms3_role_name = ParameterString(name="IamLoadFromS3RoleName", default_value="NeptuneLoadFromS3")
    
    # bulkload parameters
    bulkload_instance_type = ParameterString(name="BulkloadInstanceType", default_value="ml.m4.xlarge")
    
    # register parameters
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    deploy_instance_type = ParameterString(name="DeployInstanceType", default_value="ml.m4.xlarge")
    
    # alert parameters
    alert_emails = ParameterString(name="AlertEmails", default_value=" ")
    alert_phones = ParameterString(name="AlertPhones", default_value=" ")

    # condition parameters
    min_f1_value = ParameterFloat(name="MinF1Value", default_value=0.5)
    
    # Property files for data dependency. The output_name must be the same with those defined in steps.
    evaluation_report = PropertyFile(name="EvaluationReport", output_name="metrics", path="evaluation.json")
    neptune_metadata = PropertyFile(name="NeptuneDbMetadata", output_name="neptune_metadata", path="neptune_metadata.json")

    step_process = get_step_processing(
        bucket=default_bucket,
        region=region,
        role=role,
        params={
            'raw_input_dataset': raw_input_dataset,
            'output_dir': output_dir,
            'processing_instance_count': processing_instance_count,
            'processing_instance_type': processing_instance_type
        }
    )
    print('Step process created!')
    
    step_train = get_step_training(
        bucket=default_bucket,
        region=region,
        role=role,
        params={
            'train_instance_type': train_instance_type,
            'train_instance_count': train_instance_count,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        },
        dependencies={
            'step_process': step_process
        }
    )
    print('Step train created!')

    step_evaluate = get_step_evaluation(
        bucket=default_bucket,
        region=region,
        role=role,
        params={
            'evaluation_instance_count': evaluation_instance_count,
            'evaluation_instance_type': evaluation_instance_type
        },
        dependencies={
            'step_train': step_train,
            'step_process': step_process
        },
        properties={
            'evaluation_report': evaluation_report
        }
    )
    print('Step evaluate created!')

    step_register_model = get_step_register_model(
        model_package_group_name=model_package_group_name,
        params={
            'model_approval_status': model_approval_status,
            'transform_instance_type': transform_instance_type,
            'deploy_instance_type': deploy_instance_type
        },
        dependencies={
            'step_evaluate': step_evaluate,
            'step_train': step_train
        }
    )
    print('Step register model created!')

    step_create_model = get_step_create_model(
        bucket=default_bucket,
        region=region,
        role=role,
        sess=sess,
        params={
            'transform_model_name': transform_model_name,
            'inference_instance_type': inference_instance_type
        },
        dependencies={
            'step_train': step_train
        }
    )
    print('Step create model created!')

    step_transform = get_step_transform(
        bucket=default_bucket,
        region=region,
        role=role,
        params={
            'transform_instance_type': transform_instance_type,
            'batch_data': batch_data,
            'transform_output_prefix': transform_output_prefix
        },
        dependencies={
            'step_create_model': step_create_model
        }
    )
    print('Step transform created!')
    
    step_create_db = get_step_create_db(
        bucket=default_bucket,
        region=region,
        role=role,
        params={
            'db_cluster_identifier': db_cluster_identifier,
            'iam_loadfroms3_role_name': iam_loadfroms3_role_name
        },
        dependencies={
            'step_transform': step_transform
        },
        properties={
            'neptune_metadata': neptune_metadata
        }
    )
    print('Step create database created!')

    step_bulkload = get_step_bulkload(
        bucket=default_bucket,
        region=region,
        role=role,
        params={
            'bulkload_instance_type': bulkload_instance_type,
            'raw_input_dataset': raw_input_dataset
        },
        dependencies={
            'step_transform': step_transform,
            'step_create_db': step_create_db
        },
        properties={
            'neptune_metadata': neptune_metadata
        }
    )
    print('Step bulkload created!')
    
    step_alert = get_step_alert(
        bucket=default_bucket,
        region=region,
        role=role,
        params={
            'alert_topic': alert_topic,
            'alert_msg': alert_msg,
            'alert_emails': alert_emails,
            'alert_phones': alert_phones
        },
        dependencies={
            'step_evaluate': step_evaluate
        }
    )
    print('Step alert created!')

    step_condition = get_step_condition(
        params={
            'min_f1_value': min_f1_value
        },
        dependencies={
            'step_evaluate': step_evaluate,
            'step_register': step_register_model,
            'step_create_model': step_create_model,
            'step_transform': step_transform,
            'step_create_db': step_create_db,
            'step_bulkload': step_bulkload,
            'step_alert': step_alert
        },
        properties={
            'evaluation_report': evaluation_report
        }
    )
    print('Step condition created!')

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            raw_input_dataset,
            output_dir,
            processing_instance_count,
            processing_instance_type,
            
            train_instance_type,
            train_instance_count,
            epochs,
            learning_rate,
            batch_size,
            
            alert_topic,
            alert_msg,

            evaluation_instance_count,
            evaluation_instance_type,

            transform_model_name,
            inference_instance_type,
            
            db_cluster_identifier,
            iam_loadfroms3_role_name,

            bulkload_instance_type,
            
            transform_instance_type,
            batch_data,

            model_approval_status,
            deploy_instance_type,
            
            alert_emails,
            alert_phones,
            
            min_f1_value
        ],
        steps=[step_process, step_train, step_evaluate, step_condition],
        sagemaker_session=sess,
    )
    print('Pipeline created')
    return pipeline
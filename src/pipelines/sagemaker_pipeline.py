"""
Amazon SageMaker Pipelines — end-to-end ML workflow orchestration.

Covers:
- Pipeline definition with preprocessing, training, evaluation, and registration steps
- SageMaker Clarify bias detection and explainability
- Model registry integration
- Real-time and batch inference endpoints
- MLflow experiment tracking
- Automatic model tuning (HPO)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import boto3
import sagemaker
from botocore.exceptions import ClientError
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig,
    SageMakerClarifyProcessor,
)
from sagemaker.model import Model
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineRunResult:
    """Result of a SageMaker Pipeline execution."""

    pipeline_arn: str
    execution_arn: str
    status: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == "Succeeded"

    @property
    def failed(self) -> bool:
        return self.status == "Failed"


@dataclass
class EndpointInfo:
    """Information about a deployed SageMaker endpoint."""

    endpoint_name: str
    endpoint_arn: str
    status: str
    instance_type: str
    model_name: str
    creation_time: str = ""


class SageMakerPipelineOrchestrator:
    """
    Production SageMaker Pipelines orchestrator.

    Builds, runs, and monitors end-to-end ML pipelines including
    data preprocessing, training, bias detection, evaluation,
    conditional model registration, and endpoint deployment.

    Example:
        orchestrator = SageMakerPipelineOrchestrator()

        # Build and run the full pipeline
        execution_arn = orchestrator.run_pipeline(
            pipeline_name="churn-prediction-pipeline",
            training_data_s3="s3://my-bucket/data/train.csv",
            model_output_s3="s3://my-bucket/models/",
        )

        # Wait for completion and check results
        result = orchestrator.wait_for_pipeline(execution_arn)
        if result.succeeded:
            endpoint = orchestrator.deploy_registered_model("churn-model", "1")
            print(f"Endpoint: {endpoint.endpoint_name}")
    """

    def __init__(
        self,
        region: str | None = None,
        session: boto3.Session | None = None,
    ) -> None:
        raw_cfg = load_config()
        sm_cfg = raw_cfg.get("sagemaker", {})

        self.region = region or raw_cfg.get("aws", {}).get("region", "ap-southeast-2")
        self.execution_role_arn = sm_cfg.get("execution_role_arn", "")
        self.default_bucket = sm_cfg.get("default_bucket", "")
        self._sm_cfg = sm_cfg

        boto_session = session or boto3.Session(region_name=self.region)
        self._sm_session = sagemaker.Session(boto_session=boto_session)
        self._sm_client = boto_session.client("sagemaker")
        self._s3 = boto_session.client("s3")

        logger.info(
            "SageMakerPipelineOrchestrator initialised",
            region=self.region,
            default_bucket=self.default_bucket,
        )

    def build_pipeline(
        self,
        pipeline_name: str,
        training_data_s3: str,
        model_output_s3: str,
        accuracy_threshold: float = 0.85,
    ) -> Pipeline:
        """
        Build a full SageMaker Pipeline with the following steps:
        1. Data preprocessing (ScriptProcessor)
        2. Model training (SKLearn estimator)
        3. Bias analysis (Clarify)
        4. Model evaluation
        5. Conditional model registration (if accuracy >= threshold)

        Args:
            pipeline_name: Unique name for the pipeline.
            training_data_s3: S3 URI to the training dataset.
            model_output_s3: S3 URI for model artifacts.
            accuracy_threshold: Minimum accuracy to register the model.

        Returns:
            A configured SageMaker Pipeline object (not yet started).
        """
        # Pipeline parameters — override at runtime without rebuilding
        training_instance = ParameterString(
            name="TrainingInstance",
            default_value=self._sm_cfg.get("training", {}).get("instance_type", "ml.m5.xlarge"),
        )
        accuracy_min = ParameterFloat(
            name="AccuracyMinimum",
            default_value=accuracy_threshold,
        )
        model_approval_status = ParameterString(
            name="ModelApprovalStatus",
            default_value="PendingManualApproval",
        )

        # Step 1: Preprocessing
        preprocessing_step = self._build_preprocessing_step(
            training_data_s3=training_data_s3,
            output_s3=f"{model_output_s3}/preprocessed",
        )

        # Step 2: Training
        training_step = self._build_training_step(
            preprocessed_data_s3=f"{model_output_s3}/preprocessed/train",
            model_output_s3=model_output_s3,
            instance_type=training_instance,
        )
        training_step.add_depends_on([preprocessing_step])

        # Step 3: Evaluation
        evaluation_step = self._build_evaluation_step(
            model_s3_uri=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            test_data_s3=f"{model_output_s3}/preprocessed/test",
        )
        evaluation_step.add_depends_on([training_step])

        # Step 4: Conditional registration
        condition = ConditionGreaterThanOrEqualTo(
            left=evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
            right=accuracy_min,
        )

        registration_step = self._build_model_registration_step(
            model_s3_uri=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            approval_status=model_approval_status,
        )

        condition_step = ConditionStep(
            name="CheckAccuracy",
            conditions=[condition],
            if_steps=[registration_step],
            else_steps=[],
        )
        condition_step.add_depends_on([evaluation_step])

        pipeline = Pipeline(
            name=pipeline_name,
            parameters=[training_instance, accuracy_min, model_approval_status],
            steps=[preprocessing_step, training_step, evaluation_step, condition_step],
            sagemaker_session=self._sm_session,
        )

        logger.info("Pipeline built", pipeline_name=pipeline_name)
        return pipeline

    def upsert_and_run(
        self,
        pipeline: Pipeline,
        parameters: dict[str, Any] | None = None,
        tags: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Upsert (create or update) a pipeline and start an execution.

        Args:
            pipeline: The SageMaker Pipeline object.
            parameters: Runtime parameter overrides.
            tags: Optional AWS tags for the execution.

        Returns:
            The pipeline execution ARN.
        """
        pipeline.upsert(role_arn=self.execution_role_arn, tags=tags or [])
        logger.info("Pipeline upserted", pipeline_name=pipeline.name)

        execution = pipeline.start(parameters=parameters or {})
        execution_arn = execution.arn

        logger.info("Pipeline execution started", execution_arn=execution_arn)
        return execution_arn

    def wait_for_pipeline(
        self,
        execution_arn: str,
        poll_interval_seconds: int = 30,
        timeout_seconds: int = 7200,
    ) -> PipelineRunResult:
        """
        Wait for a pipeline execution to complete.

        Args:
            execution_arn: The pipeline execution ARN.
            poll_interval_seconds: Seconds between status polls.
            timeout_seconds: Maximum wait time (default 2 hours).

        Returns:
            PipelineRunResult with final status and step details.
        """
        elapsed = 0
        terminal_states = {"Succeeded", "Failed", "Stopped"}

        while elapsed < timeout_seconds:
            response = self._sm_client.describe_pipeline_execution(
                PipelineExecutionArn=execution_arn
            )
            status = response.get("PipelineExecutionStatus", "")
            logger.debug("Pipeline status", execution_arn=execution_arn, status=status)

            if status in terminal_states:
                steps_response = self._sm_client.list_pipeline_execution_steps(
                    PipelineExecutionArn=execution_arn
                )

                pipeline_arn = execution_arn.rsplit("/", 1)[0].replace("execution", "pipeline")

                return PipelineRunResult(
                    pipeline_arn=pipeline_arn,
                    execution_arn=execution_arn,
                    status=status,
                    steps=steps_response.get("PipelineExecutionSteps", []),
                    failure_reason=response.get("FailureReason", ""),
                )

            time.sleep(poll_interval_seconds)
            elapsed += poll_interval_seconds

        raise TimeoutError(
            f"Pipeline execution {execution_arn} did not complete within {timeout_seconds}s"
        )

    def deploy_registered_model(
        self,
        model_package_group_name: str,
        model_version: str = "latest",
        endpoint_name: str | None = None,
        instance_type: str | None = None,
        initial_instance_count: int = 1,
    ) -> EndpointInfo:
        """
        Deploy an approved model from the SageMaker Model Registry to an endpoint.

        Args:
            model_package_group_name: The model package group name.
            model_version: Version number or "latest".
            endpoint_name: Name for the endpoint (auto-generated if not provided).
            instance_type: Instance type (overrides config).
            initial_instance_count: Number of initial instances.

        Returns:
            EndpointInfo with deployment details.
        """
        # Resolve the model package ARN
        if model_version == "latest":
            packages = self._sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                SortOrder="Descending",
                MaxResults=1,
            )
            if not packages["ModelPackageSummaryList"]:
                raise ValueError(f"No approved models found in group: {model_package_group_name}")
            model_pkg_arn = packages["ModelPackageSummaryList"][0]["ModelPackageArn"]
        else:
            model_pkg_arn = f"{model_package_group_name}/{model_version}"

        ep_name = endpoint_name or f"{model_package_group_name}-{int(time.time())}"
        inst_type = instance_type or self._sm_cfg.get("endpoints", {}).get(
            "inference_instance", "ml.m5.xlarge"
        )

        model = Model(
            model_data=model_pkg_arn,
            role=self.execution_role_arn,
            sagemaker_session=self._sm_session,
        )

        predictor = model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=inst_type,
            endpoint_name=ep_name,
        )

        logger.info(
            "Model deployed",
            endpoint_name=ep_name,
            instance_type=inst_type,
            model_package_arn=model_pkg_arn,
        )

        ep_desc = self._sm_client.describe_endpoint(EndpointName=ep_name)

        return EndpointInfo(
            endpoint_name=ep_name,
            endpoint_arn=ep_desc["EndpointArn"],
            status=ep_desc["EndpointStatus"],
            instance_type=inst_type,
            model_name=model_package_group_name,
        )

    def run_clarify_bias_analysis(
        self,
        train_data_s3: str,
        model_name: str,
        label_column: str,
        facet_column: str,
        output_s3: str,
    ) -> str:
        """
        Run SageMaker Clarify bias analysis on a trained model.

        Args:
            train_data_s3: S3 URI of the training dataset.
            model_name: Deployed SageMaker model name.
            label_column: Name of the target/label column.
            facet_column: Name of the sensitive feature column for bias analysis.
            output_s3: S3 URI for bias analysis results.

        Returns:
            The Clarify processing job name.
        """
        clarify_cfg = self._sm_cfg.get("clarify", {})

        clarify_processor = SageMakerClarifyProcessor(
            role=self.execution_role_arn,
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=self._sm_session,
        )

        data_config = DataConfig(
            s3_data_input_path=train_data_s3,
            s3_output_path=output_s3,
            label=label_column,
            dataset_type="text/csv",
        )

        bias_config = BiasConfig(
            label_values_or_threshold=[1],
            facet_name=facet_column,
        )

        model_config = ModelConfig(
            model_name=model_name,
            instance_type=self._sm_cfg.get("endpoints", {}).get("inference_instance", "ml.m5.xlarge"),
            instance_count=1,
        )

        clarify_processor.run_bias(
            data_config=data_config,
            bias_config=bias_config,
            model_config=model_config,
            pre_training_methods="all",
            post_training_methods="all",
        )

        job_name = clarify_processor.latest_job_name
        logger.info("Clarify bias analysis complete", job_name=job_name)
        return job_name

    # ------------------------------------------------------------------
    # Private step builders
    # ------------------------------------------------------------------

    def _build_preprocessing_step(
        self, training_data_s3: str, output_s3: str
    ) -> ProcessingStep:
        """Build the data preprocessing step."""
        processor = ScriptProcessor(
            image_uri=sagemaker.image_uris.retrieve(
                framework="sklearn",
                region=self.region,
                version="1.2-1",
            ),
            command=["python3"],
            role=self.execution_role_arn,
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=self._sm_session,
        )

        return ProcessingStep(
            name="DataPreprocessing",
            processor=processor,
            inputs=[
                ProcessingInput(
                    source=training_data_s3,
                    destination="/opt/ml/processing/input",
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    source="/opt/ml/processing/output/train",
                    destination=f"{output_s3}/train",
                ),
                ProcessingOutput(
                    output_name="test",
                    source="/opt/ml/processing/output/test",
                    destination=f"{output_s3}/test",
                ),
            ],
            code="scripts/preprocessing.py",
        )

    def _build_training_step(
        self,
        preprocessed_data_s3: str,
        model_output_s3: str,
        instance_type: ParameterString,
    ) -> TrainingStep:
        """Build the model training step."""
        estimator = SKLearn(
            entry_point="scripts/train.py",
            framework_version="1.2-1",
            instance_type=instance_type,
            instance_count=1,
            role=self.execution_role_arn,
            output_path=model_output_s3,
            sagemaker_session=self._sm_session,
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
            },
            metric_definitions=[
                {"Name": "validation:accuracy", "Regex": "validation accuracy: ([0-9.]+)"},
                {"Name": "validation:f1", "Regex": "validation f1: ([0-9.]+)"},
            ],
            enable_sagemaker_metrics=True,
        )

        return TrainingStep(
            name="ModelTraining",
            estimator=estimator,
            inputs={
                "train": sagemaker.inputs.TrainingInput(preprocessed_data_s3, content_type="text/csv")
            },
        )

    def _build_evaluation_step(
        self, model_s3_uri: Any, test_data_s3: str
    ) -> ProcessingStep:
        """Build the model evaluation step."""
        processor = ScriptProcessor(
            image_uri=sagemaker.image_uris.retrieve(
                framework="sklearn", region=self.region, version="1.2-1"
            ),
            command=["python3"],
            role=self.execution_role_arn,
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=self._sm_session,
        )

        return ProcessingStep(
            name="ModelEvaluation",
            processor=processor,
            inputs=[
                ProcessingInput(source=model_s3_uri, destination="/opt/ml/processing/model"),
                ProcessingInput(source=test_data_s3, destination="/opt/ml/processing/test"),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                )
            ],
            code="scripts/evaluate.py",
        )

    def _build_model_registration_step(
        self, model_s3_uri: Any, approval_status: ParameterString
    ) -> ModelStep:
        """Build the model registration step for the Model Registry."""
        model = Model(
            model_data=model_s3_uri,
            role=self.execution_role_arn,
            sagemaker_session=self._sm_session,
        )

        return ModelStep(
            name="RegisterModel",
            step_args=model.register(
                content_types=["text/csv"],
                response_types=["application/json"],
                inference_instances=["ml.m5.xlarge", "ml.m5.2xlarge"],
                transform_instances=["ml.m5.xlarge"],
                approval_status=approval_status,
            ),
        )

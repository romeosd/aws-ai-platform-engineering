"""
Amazon Bedrock Model Evaluation — automated LLM quality assessment.

Covers:
- Built-in evaluation jobs (accuracy, robustness, toxicity)
- Custom evaluation with human workers (A2I integration)
- RAG evaluation (faithfulness, answer relevance, context precision)
- Comparison between model versions
- CloudWatch metric publishing
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetric:
    """A single evaluation metric result."""

    name: str
    score: float
    category: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGEvaluationResult:
    """Result of evaluating a RAG pipeline response."""

    question: str
    generated_answer: str
    reference_answer: str
    retrieved_contexts: list[str]

    faithfulness: float = 0.0          # Is the answer grounded in the retrieved context?
    answer_relevance: float = 0.0      # Does the answer address the question?
    context_precision: float = 0.0     # Are the retrieved chunks relevant to the question?
    context_recall: float = 0.0        # Does context cover what's needed to answer?

    overall_score: float = 0.0
    evaluator_reasoning: str = ""

    def __post_init__(self) -> None:
        scores = [
            s for s in [
                self.faithfulness,
                self.answer_relevance,
                self.context_precision,
                self.context_recall,
            ] if s > 0
        ]
        self.overall_score = sum(scores) / len(scores) if scores else 0.0

    def passed(self, threshold: float = 0.7) -> bool:
        return self.overall_score >= threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "overall_score": self.overall_score,
            "passed": self.passed(),
        }


class BedrockModelEvaluator:
    """
    Bedrock Model Evaluation client.

    Runs automated and human-in-the-loop evaluation jobs
    for foundation models and RAG pipelines.

    Example:
        evaluator = BedrockModelEvaluator()

        # Evaluate RAG pipeline quality
        result = evaluator.evaluate_rag_response(
            question="What is the data retention policy?",
            generated_answer=rag_response.answer,
            reference_answer="Data is retained for 7 years per policy 4.2.1",
            retrieved_contexts=[chunk.content for chunk in chunks],
        )

        print(f"Overall score: {result.overall_score:.2f}")
        print(f"Faithfulness: {result.faithfulness:.2f}")
        if not result.passed():
            print("RAG response below quality threshold")
    """

    def __init__(
        self,
        region: str | None = None,
        session: boto3.Session | None = None,
    ) -> None:
        raw_cfg = load_config()
        self.region = region or raw_cfg.get("aws", {}).get("region", "ap-southeast-2")
        self._bedrock_client = (session or boto3.Session()).client("bedrock", region_name=self.region)
        self._bedrock_runtime = (session or boto3.Session()).client("bedrock-runtime", region_name=self.region)
        self._cloudwatch = (session or boto3.Session()).client("cloudwatch", region_name=self.region)
        self._s3 = (session or boto3.Session()).client("s3", region_name=self.region)

        self._eval_model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"

        logger.info("BedrockModelEvaluator initialised", region=self.region)

    def evaluate_rag_response(
        self,
        question: str,
        generated_answer: str,
        reference_answer: str,
        retrieved_contexts: list[str],
    ) -> RAGEvaluationResult:
        """
        Evaluate a RAG pipeline response using LLM-as-judge approach.

        Metrics evaluated:
        - Faithfulness: Is the answer grounded in retrieved context?
        - Answer Relevance: Does the answer address the question?
        - Context Precision: Are retrieved chunks relevant?
        - Context Recall: Do chunks cover what's needed?

        Args:
            question: The original user question.
            generated_answer: The RAG pipeline's answer.
            reference_answer: The expected ground-truth answer.
            retrieved_contexts: List of retrieved context chunks.

        Returns:
            RAGEvaluationResult with per-metric scores and overall quality.
        """
        context_text = "\n\n".join(
            f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(retrieved_contexts)
        )

        eval_prompt = f"""You are an expert RAG system evaluator. Evaluate the following RAG response.

QUESTION: {question}

RETRIEVED CONTEXTS:
{context_text}

GENERATED ANSWER: {generated_answer}

REFERENCE ANSWER: {reference_answer}

Evaluate the response on these four metrics, each scored 0.0 to 1.0:

1. FAITHFULNESS (0.0-1.0): Is every claim in the generated answer directly supported by the retrieved contexts?
   - 1.0 = Every claim is directly grounded in context
   - 0.5 = Some claims are grounded, some are not
   - 0.0 = The answer contradicts or fabricates beyond context

2. ANSWER_RELEVANCE (0.0-1.0): Does the generated answer directly address the question?
   - 1.0 = Directly and completely answers the question
   - 0.5 = Partially answers the question
   - 0.0 = Does not address the question

3. CONTEXT_PRECISION (0.0-1.0): Are the retrieved contexts relevant to answering the question?
   - 1.0 = All retrieved contexts are highly relevant
   - 0.5 = Mix of relevant and irrelevant contexts
   - 0.0 = Retrieved contexts are not relevant to the question

4. CONTEXT_RECALL (0.0-1.0): Do the retrieved contexts contain enough information to answer the question?
   - 1.0 = Contexts contain all necessary information
   - 0.5 = Contexts contain some but not all needed information
   - 0.0 = Contexts lack information needed to answer

Respond ONLY with valid JSON in this exact format:
{{
  "faithfulness": <float>,
  "answer_relevance": <float>,
  "context_precision": <float>,
  "context_recall": <float>,
  "reasoning": "<brief explanation of each score>"
}}"""

        try:
            response = self._bedrock_runtime.invoke_model(
                modelId=self._eval_model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": eval_prompt}],
                }),
                contentType="application/json",
                accept="application/json",
            )
            raw = json.loads(response["body"].read())
            eval_text = raw["content"][0]["text"]
            scores = json.loads(eval_text)
        except Exception as exc:
            logger.error("RAG evaluation failed", error=str(exc))
            return RAGEvaluationResult(
                question=question,
                generated_answer=generated_answer,
                reference_answer=reference_answer,
                retrieved_contexts=retrieved_contexts,
            )

        result = RAGEvaluationResult(
            question=question,
            generated_answer=generated_answer,
            reference_answer=reference_answer,
            retrieved_contexts=retrieved_contexts,
            faithfulness=float(scores.get("faithfulness", 0.0)),
            answer_relevance=float(scores.get("answer_relevance", 0.0)),
            context_precision=float(scores.get("context_precision", 0.0)),
            context_recall=float(scores.get("context_recall", 0.0)),
            evaluator_reasoning=scores.get("reasoning", ""),
        )

        logger.info(
            "RAG evaluation complete",
            overall_score=f"{result.overall_score:.2f}",
            faithfulness=f"{result.faithfulness:.2f}",
            answer_relevance=f"{result.answer_relevance:.2f}",
            passed=result.passed(),
        )

        return result

    def evaluate_batch(
        self,
        evaluation_pairs: list[dict[str, Any]],
        publish_to_cloudwatch: bool = True,
    ) -> list[RAGEvaluationResult]:
        """
        Evaluate a batch of RAG responses.

        Args:
            evaluation_pairs: List of dicts with keys:
                question, generated_answer, reference_answer, retrieved_contexts
            publish_to_cloudwatch: Whether to publish aggregate metrics to CloudWatch.

        Returns:
            List of RAGEvaluationResult, one per pair.
        """
        results: list[RAGEvaluationResult] = []

        for i, pair in enumerate(evaluation_pairs):
            logger.info("Evaluating batch item", index=i, total=len(evaluation_pairs))
            result = self.evaluate_rag_response(**pair)
            results.append(result)

        if publish_to_cloudwatch and results:
            self._publish_evaluation_metrics(results)

        passed = sum(1 for r in results if r.passed())
        logger.info(
            "Batch evaluation complete",
            total=len(results),
            passed=passed,
            pass_rate=f"{passed/len(results)*100:.1f}%",
        )

        return results

    def start_bedrock_evaluation_job(
        self,
        job_name: str,
        model_id: str,
        dataset_s3_uri: str,
        output_s3_uri: str,
        role_arn: str,
        task_type: str = "Summarization",
    ) -> str:
        """
        Start a native Bedrock Model Evaluation job.

        Args:
            job_name: Unique name for the evaluation job.
            model_id: The Bedrock model ARN to evaluate.
            dataset_s3_uri: S3 URI to the evaluation dataset JSONL.
            output_s3_uri: S3 URI for evaluation results.
            role_arn: IAM role ARN with Bedrock and S3 permissions.
            task_type: Summarization | QuestionAndAnswer | Classification | Custom

        Returns:
            The evaluation job ARN.
        """
        try:
            response = self._bedrock_client.create_evaluation_job(
                jobName=job_name,
                jobDescription=f"Automated evaluation job for {model_id}",
                roleArn=role_arn,
                customerEncryptionKeyId=None,
                evaluationConfig={
                    "automated": {
                        "datasetMetricConfigs": [
                            {
                                "taskType": task_type,
                                "dataset": {
                                    "name": job_name,
                                    "datasetLocation": {"s3Uri": dataset_s3_uri},
                                },
                                "metricNames": [
                                    "Builtin.Accuracy",
                                    "Builtin.Robustness",
                                    "Builtin.Toxicity",
                                ],
                            }
                        ]
                    }
                },
                inferenceConfig={
                    "models": [
                        {
                            "bedrockModel": {
                                "modelIdentifier": model_id,
                                "inferenceParams": json.dumps({"temperature": 0.0, "max_tokens": 1024}),
                            }
                        }
                    ]
                },
                outputDataConfig={"s3Uri": output_s3_uri},
            )
        except ClientError as exc:
            logger.error("Failed to start Bedrock evaluation job", error=str(exc))
            raise

        job_arn = response["jobArn"]
        logger.info("Bedrock evaluation job started", job_arn=job_arn, job_name=job_name)
        return job_arn

    def _publish_evaluation_metrics(self, results: list[RAGEvaluationResult]) -> None:
        """Publish aggregate evaluation metrics to CloudWatch."""
        if not results:
            return

        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_relevance = sum(r.answer_relevance for r in results) / len(results)
        avg_precision = sum(r.context_precision for r in results) / len(results)
        avg_recall = sum(r.context_recall for r in results) / len(results)
        avg_overall = sum(r.overall_score for r in results) / len(results)
        pass_rate = sum(1 for r in results if r.passed()) / len(results)

        metric_data = [
            {"MetricName": "RAG/Faithfulness", "Value": avg_faithfulness, "Unit": "None"},
            {"MetricName": "RAG/AnswerRelevance", "Value": avg_relevance, "Unit": "None"},
            {"MetricName": "RAG/ContextPrecision", "Value": avg_precision, "Unit": "None"},
            {"MetricName": "RAG/ContextRecall", "Value": avg_recall, "Unit": "None"},
            {"MetricName": "RAG/OverallScore", "Value": avg_overall, "Unit": "None"},
            {"MetricName": "RAG/PassRate", "Value": pass_rate, "Unit": "None"},
        ]

        try:
            self._cloudwatch.put_metric_data(
                Namespace="AWSAIPlatform",
                MetricData=metric_data,
            )
            logger.info("Evaluation metrics published to CloudWatch", metric_count=len(metric_data))
        except ClientError as exc:
            logger.warning("Failed to publish metrics to CloudWatch", error=str(exc))

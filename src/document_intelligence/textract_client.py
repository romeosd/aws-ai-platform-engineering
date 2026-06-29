"""
Amazon Textract — document intelligence client.

Provides:
- Synchronous document analysis (images up to 10MB)
- Asynchronous document analysis (PDFs, large files via S3)
- Table extraction with structured output
- Form field key-value extraction
- Layout analysis (headings, paragraphs, page structure)
- Signature detection
- Expense and invoice analysis
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TextractTable:
    """A table extracted from a document."""

    page: int
    rows: list[list[str]] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0

    def __post_init__(self) -> None:
        self.row_count = len(self.rows)
        self.column_count = max((len(r) for r in self.rows), default=0)

    def to_markdown(self) -> str:
        """Render the table as a Markdown string."""
        if not self.rows:
            return ""
        header = "| " + " | ".join(self.rows[0]) + " |"
        separator = "| " + " | ".join(["---"] * len(self.rows[0])) + " |"
        body = "\n".join("| " + " | ".join(row) + " |" for row in self.rows[1:])
        return "\n".join([header, separator, body])


@dataclass
class FormField:
    """A key-value form field extracted from a document."""

    key: str
    value: str
    page: int
    confidence: float = 0.0


@dataclass
class TextractResult:
    """Aggregated result from a Textract document analysis."""

    raw_text: str
    pages: int = 0
    tables: list[TextractTable] = field(default_factory=list)
    form_fields: list[FormField] = field(default_factory=list)
    layout_blocks: list[dict[str, Any]] = field(default_factory=list)
    signatures_detected: int = 0
    raw_blocks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def table_count(self) -> int:
        return len(self.tables)

    @property
    def form_field_count(self) -> int:
        return len(self.form_fields)

    def get_form_value(self, key: str, case_sensitive: bool = False) -> str | None:
        """Look up a form field value by key name."""
        for field in self.form_fields:
            field_key = field.key if case_sensitive else field.key.lower()
            search_key = key if case_sensitive else key.lower()
            if field_key == search_key:
                return field.value
        return None

    def tables_as_markdown(self) -> str:
        """Return all tables formatted as Markdown."""
        parts = []
        for i, table in enumerate(self.tables):
            parts.append(f"### Table {i+1} (Page {table.page})\n{table.to_markdown()}")
        return "\n\n".join(parts)


class TextractClient:
    """
    Production Amazon Textract client.

    Handles both synchronous (images) and asynchronous (PDFs via S3)
    document analysis, with structured extraction of text, tables,
    forms, and layout.

    Example:
        textract = TextractClient()

        # Analyse a local image synchronously
        result = textract.analyse_document_sync(Path("invoice.png"))
        print(result.raw_text)
        print(result.tables_as_markdown())

        # Upload a PDF to S3 and analyse asynchronously
        job_id = textract.start_document_analysis_s3(
            s3_bucket="my-bucket",
            s3_key="documents/contract.pdf",
        )
        result = textract.wait_for_job(job_id)
        print(result.form_fields)
    """

    def __init__(
        self,
        region: str | None = None,
        session: boto3.Session | None = None,
    ) -> None:
        raw_cfg = load_config()
        self.region = region or raw_cfg.get("aws", {}).get("region", "ap-southeast-2")
        self._cfg = raw_cfg.get("textract", {})

        self._session = session or boto3.Session()
        self._client = self._session.client("textract", region_name=self.region)
        self._s3 = self._session.client("s3", region_name=self.region)

        logger.info("TextractClient initialised", region=self.region)

    def analyse_document_sync(
        self,
        file_path: Path,
        feature_types: list[str] | None = None,
    ) -> TextractResult:
        """
        Analyse a document image synchronously.

        Supports JPEG, PNG, TIFF up to 10MB.
        Use start_document_analysis_s3 for larger files and PDFs.

        Args:
            file_path: Path to the document image.
            feature_types: Textract features to extract.
                          Defaults to config value (TABLES, FORMS, SIGNATURES, LAYOUT).

        Returns:
            TextractResult with extracted text, tables, and forms.
        """
        features = feature_types or self._cfg.get(
            "feature_types", ["TABLES", "FORMS", "SIGNATURES", "LAYOUT"]
        )

        image_bytes = Path(file_path).read_bytes()

        try:
            response = self._client.analyze_document(
                Document={"Bytes": image_bytes},
                FeatureTypes=features,
            )
        except ClientError as exc:
            logger.error("Textract sync analysis failed", error=str(exc), file=str(file_path))
            raise

        blocks = response.get("Blocks", [])
        logger.info(
            "Textract sync analysis complete",
            file=str(file_path),
            block_count=len(blocks),
        )

        return self._parse_blocks(blocks)

    def start_document_analysis_s3(
        self,
        s3_bucket: str,
        s3_key: str,
        feature_types: list[str] | None = None,
        output_bucket: str | None = None,
        output_prefix: str = "textract-output/",
        notification_channel: dict[str, str] | None = None,
    ) -> str:
        """
        Start an asynchronous Textract job for S3-hosted documents.

        Args:
            s3_bucket: S3 bucket containing the document.
            s3_key: S3 key (path) to the document.
            feature_types: Textract features to extract.
            output_bucket: S3 bucket for results (defaults to config).
            output_prefix: S3 prefix for results.
            notification_channel: Optional SNS notification config.

        Returns:
            Textract job ID string.
        """
        features = feature_types or self._cfg.get(
            "feature_types", ["TABLES", "FORMS", "SIGNATURES", "LAYOUT"]
        )
        out_bucket = output_bucket or self._cfg.get("output_bucket", s3_bucket)

        request: dict[str, Any] = {
            "DocumentLocation": {
                "S3Object": {"Bucket": s3_bucket, "Name": s3_key}
            },
            "FeatureTypes": features,
            "OutputConfig": {
                "S3Bucket": out_bucket,
                "S3Prefix": output_prefix,
            },
        }

        role_arn = self._cfg.get("async_role_arn", "")
        if role_arn:
            request["NotificationChannel"] = {
                "SNSTopicArn": self._cfg.get("sns_topic_arn", ""),
                "RoleArn": role_arn,
            }

        if notification_channel:
            request["NotificationChannel"] = notification_channel

        try:
            response = self._client.start_document_analysis(**request)
        except ClientError as exc:
            logger.error("Failed to start async Textract job", error=str(exc))
            raise

        job_id = response["JobId"]
        logger.info(
            "Textract async job started",
            job_id=job_id,
            s3_path=f"s3://{s3_bucket}/{s3_key}",
        )
        return job_id

    def wait_for_job(
        self,
        job_id: str,
        poll_interval_seconds: int = 5,
        timeout_seconds: int = 600,
    ) -> TextractResult:
        """
        Poll until a Textract async job completes and return its results.

        Args:
            job_id: The Textract job ID returned by start_document_analysis_s3.
            poll_interval_seconds: Seconds between status checks.
            timeout_seconds: Maximum wait time before raising TimeoutError.

        Returns:
            TextractResult with all extracted content.

        Raises:
            TimeoutError: If the job does not complete within timeout_seconds.
            RuntimeError: If the job fails.
        """
        elapsed = 0
        while elapsed < timeout_seconds:
            response = self._client.get_document_analysis(JobId=job_id)
            status = response["JobStatus"]

            logger.debug("Textract job status", job_id=job_id, status=status)

            if status == "SUCCEEDED":
                all_blocks: list[dict[str, Any]] = list(response.get("Blocks", []))

                # Paginate if there are more blocks
                while "NextToken" in response:
                    response = self._client.get_document_analysis(
                        JobId=job_id, NextToken=response["NextToken"]
                    )
                    all_blocks.extend(response.get("Blocks", []))

                logger.info(
                    "Textract async job complete",
                    job_id=job_id,
                    block_count=len(all_blocks),
                )
                return self._parse_blocks(all_blocks)

            elif status == "FAILED":
                error = response.get("StatusMessage", "Unknown error")
                raise RuntimeError(f"Textract job {job_id} failed: {error}")

            time.sleep(poll_interval_seconds)
            elapsed += poll_interval_seconds

        raise TimeoutError(
            f"Textract job {job_id} did not complete within {timeout_seconds}s"
        )

    def analyse_expense(self, s3_bucket: str, s3_key: str) -> dict[str, Any]:
        """
        Run Textract Expense Analysis on an invoice or receipt.

        Returns raw expense fields including vendor, total, line items.
        """
        response = self._client.start_expense_analysis(
            DocumentLocation={"S3Object": {"Bucket": s3_bucket, "Name": s3_key}}
        )
        job_id = response["JobId"]
        logger.info("Expense analysis job started", job_id=job_id)

        # Poll for completion
        while True:
            result = self._client.get_expense_analysis(JobId=job_id)
            if result["JobStatus"] == "SUCCEEDED":
                return result
            elif result["JobStatus"] == "FAILED":
                raise RuntimeError(f"Expense analysis job {job_id} failed")
            time.sleep(3)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_blocks(self, blocks: list[dict[str, Any]]) -> TextractResult:
        """Convert raw Textract block output into a TextractResult."""
        block_map: dict[str, dict[str, Any]] = {b["Id"]: b for b in blocks}

        raw_text_parts: list[str] = []
        tables: list[TextractTable] = []
        form_fields: list[FormField] = []
        layout_blocks: list[dict[str, Any]] = []
        signatures = 0
        pages = 0

        for block in blocks:
            btype = block.get("BlockType", "")

            if btype == "PAGE":
                pages += 1

            elif btype == "LINE":
                raw_text_parts.append(block.get("Text", ""))

            elif btype == "TABLE":
                table = self._extract_table(block, block_map)
                tables.append(table)

            elif btype == "KEY_VALUE_SET" and block.get("EntityTypes", []) == ["KEY"]:
                field = self._extract_form_field(block, block_map)
                if field:
                    form_fields.append(field)

            elif btype == "SIGNATURE":
                signatures += 1

            elif btype.startswith("LAYOUT_"):
                layout_blocks.append({
                    "type": btype,
                    "page": block.get("Page", 0),
                    "text": block.get("Text", ""),
                })

        return TextractResult(
            raw_text="\n".join(raw_text_parts),
            pages=pages,
            tables=tables,
            form_fields=form_fields,
            layout_blocks=layout_blocks,
            signatures_detected=signatures,
            raw_blocks=blocks,
        )

    def _extract_table(
        self, table_block: dict[str, Any], block_map: dict[str, dict[str, Any]]
    ) -> TextractTable:
        """Extract a structured table from Textract blocks."""
        rows: dict[int, dict[int, str]] = {}

        for rel in table_block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for cell_id in rel["Ids"]:
                    cell = block_map.get(cell_id, {})
                    if cell.get("BlockType") == "CELL":
                        row_idx = cell.get("RowIndex", 0)
                        col_idx = cell.get("ColumnIndex", 0)
                        cell_text = self._get_cell_text(cell, block_map)
                        rows.setdefault(row_idx, {})[col_idx] = cell_text

        sorted_rows = [
            [rows[r].get(c, "") for c in sorted(rows[r])]
            for r in sorted(rows)
        ]

        return TextractTable(
            page=table_block.get("Page", 0),
            rows=sorted_rows,
        )

    def _get_cell_text(
        self, cell_block: dict[str, Any], block_map: dict[str, dict[str, Any]]
    ) -> str:
        """Extract text content from a table cell block."""
        parts: list[str] = []
        for rel in cell_block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for word_id in rel["Ids"]:
                    word = block_map.get(word_id, {})
                    if word.get("BlockType") == "WORD":
                        parts.append(word.get("Text", ""))
        return " ".join(parts)

    def _extract_form_field(
        self, key_block: dict[str, Any], block_map: dict[str, dict[str, Any]]
    ) -> FormField | None:
        """Extract a key-value form field from Textract blocks."""
        key_text = self._get_cell_text(key_block, block_map)
        value_text = ""

        for rel in key_block.get("Relationships", []):
            if rel["Type"] == "VALUE":
                for val_id in rel["Ids"]:
                    val_block = block_map.get(val_id, {})
                    value_text = self._get_cell_text(val_block, block_map)

        if not key_text:
            return None

        return FormField(
            key=key_text.strip(),
            value=value_text.strip(),
            page=key_block.get("Page", 0),
            confidence=key_block.get("Confidence", 0.0),
        )

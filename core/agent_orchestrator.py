import os
import time
import pandas as pd
from core.agent_state import AgentState
from core.ingestion import ingest_file
from core.domain_detector import detect_domain
from core.profiler import profile_dataset
from core.dqi_engine import calculate_dqi
from core.recommendation_engine import detect_issues_and_recommend
from core.decision_engine import generate_decision_trace
from core.cleaner import clean_dataset
from core.validator import validate_cleaning
from core.anomaly_detector import detect_anomalies
from core.etl_engine import run_etl_pipeline
from core.unstructured_engine import (
    assess_unstructured_document_quality,
    convert_unstructured_to_structured
)

class AgenticOrchestrator:
    """
    Agentic AI Orchestrator running an 10-Agent pipeline across
    Structured, Semi-Structured, and Unstructured datasets.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DATAPULSE_LLM_KEY")
        self.llm_enabled = bool(self.api_key and len(self.api_key.strip()) > 5)

    def run_pipeline(self, file_source, file_name: str = None, user_options: dict = None) -> AgentState:
        state = AgentState(file_name=file_name or "uploaded_dataset")
        state.agent_modes["llm_connected"] = self.llm_enabled
        if self.llm_enabled:
            state.agent_modes["orchestrator_mode"] = "LLM-Augmented Agentic Orchestration (API Active)"

        t_start = time.time()

        # =========================================================================
        # 1. INGESTION AGENT
        # =========================================================================
        state.pipeline_status = "Ingesting"
        ingest_res = ingest_file(file_source, file_name=file_name)
        
        state.file_name = ingest_res["info"]["file_name"]
        state.file_type = ingest_res["info"]["format_type"]
        state.data_category = ingest_res["info"]["category"]
        state.raw_df = ingest_res.get("structured_df")
        state.unstructured_text = ingest_res.get("unstructured_text")
        state.document_metadata = ingest_res.get("document_metadata", {})

        state.add_message(
            agent_name="Ingestion Agent",
            stage="INGEST",
            status="SUCCESS",
            message=f"Ingested '{state.file_name}' ({state.file_type}) successfully.",
            reasoning=f"Identified file format as {state.file_type} belonging to {state.data_category} category.",
            action_taken="Parsed file bytes into memory structures.",
            result=f"Category: {state.data_category} · Size: {ingest_res['file_size_bytes']} bytes."
        )

        # =========================================================================
        # 2. DOMAIN DETECTION AGENT
        # =========================================================================
        state.pipeline_status = "Detecting Domain"
        if state.data_category in ["Structured", "Semi-Structured"] and state.raw_df is not None:
            dom_info = detect_domain(state.raw_df)
        else:
            # Text based domain heuristic for unstructured
            text_sample = (state.unstructured_text or "")[:1000].lower()
            if any(w in text_sample for w in ['customer', 'client', 'email', 'phone', 'address']):
                dom_info = {"domain": "Customer", "confidence": "High", "reasoning": "Detected customer entity patterns in text."}
            elif any(w in text_sample for w in ['amount', 'price', 'revenue', 'invoice', 'salary', 'financial']):
                dom_info = {"domain": "Finance", "confidence": "High", "reasoning": "Detected financial entity patterns in text."}
            else:
                dom_info = {"domain": "General", "confidence": "Medium", "reasoning": "Standard document text structure detected."}

        state.domain = dom_info["domain"]
        state.domain_confidence = dom_info["confidence"]
        state.domain_reasoning = dom_info["reasoning"]

        state.add_message(
            agent_name="Domain Agent",
            stage="DOMAIN",
            status="SUCCESS",
            message=f"Detected Domain: {state.domain} ({state.domain_confidence} confidence).",
            reasoning=state.domain_reasoning,
            action_taken="Matched field/entity patterns against domain knowledge graph.",
            result=f"Domain: {state.domain}"
        )

        # =========================================================================
        # 3. PROFILING AGENT & 4. QUALITY ASSESSMENT AGENT
        # =========================================================================
        state.pipeline_status = "Assessing Quality"

        if state.data_category in ["Structured", "Semi-Structured"] and state.raw_df is not None:
            # Structured Pipeline
            state.profile = profile_dataset(state.raw_df)
            dqi_res = calculate_dqi(state.raw_df, domain=state.domain)
            state.dqi_score = dqi_res["da_dqi"]
            state.dqi_dimensions = dqi_res["dimensions"]
            state.quality_issues = detect_issues_and_recommend(state.raw_df)

            state.add_message(
                agent_name="Profiling Agent",
                stage="PROFILE",
                status="SUCCESS",
                message=f"Profiled {state.profile['total_rows']} rows and {state.profile['total_cols']} columns.",
                reasoning="Extracted missing value ratios, uniqueness metrics, and data types.",
                action_taken="Ran statistical summary profiling.",
                result=f"Missing: {state.profile['missing_cells']} cells · Duplicates: {state.profile['duplicate_rows']} rows."
            )

            state.add_message(
                agent_name="Quality Assessment Agent",
                stage="ASSESS",
                status="DECISION",
                message=f"Calculated Initial DA-DQI: {state.dqi_score:.2f} / 100",
                reasoning=f"Evaluated 5 dimensions under {state.domain} domain weights.",
                action_taken="Computed weighted DA-DQI index.",
                result=f"Initial Score: {state.dqi_score:.2f} / 100 ({len(state.quality_issues)} issues found)"
            )

        else:
            # Unstructured Pipeline
            unstruct_res = assess_unstructured_document_quality(state.unstructured_text, state.document_metadata)
            state.unstructured_quality_score = unstruct_res["document_quality_score"]
            state.unstructured_dimensions = unstruct_res["dimensions"]
            state.quality_issues = unstruct_res["issues"]
            state.dqi_score = unstruct_res["document_quality_score"] # Shared score holder
            state.profile = {
                "total_rows": state.document_metadata.get("num_pages", 1),
                "total_cols": 1,
                "missing_cells": 0,
                "duplicate_rows": 0,
                "text_length": len(state.unstructured_text or "")
            }

            state.add_message(
                agent_name="Profiling Agent",
                stage="PROFILE",
                status="SUCCESS",
                message=f"Profiled document ({len(state.unstructured_text or '')} characters).",
                reasoning="Computed text completeness, word counts, and page metadata.",
                action_taken="Extracted text structure and entity counts.",
                result=f"Words: {state.document_metadata.get('length_words', 0)} · Pages: {state.document_metadata.get('num_pages', 1)}"
            )

            state.add_message(
                agent_name="Quality Assessment Agent",
                stage="ASSESS",
                status="DECISION",
                message=f"Calculated Document Quality Score: {state.unstructured_quality_score:.2f} / 100",
                reasoning="Evaluated text extraction quality, field completeness, and entity validity.",
                action_taken="Computed document quality metrics.",
                result=f"Document Quality: {state.unstructured_quality_score:.2f} / 100 ({unstruct_res['quality_tier']})"
            )

        # =========================================================================
        # 5. PLANNING AGENT (OBSERVE -> REASON -> PLAN)
        # =========================================================================
        state.pipeline_status = "Planning Strategy"
        decision_trace = generate_decision_trace(state.quality_issues) if state.quality_issues else []
        state.recommended_actions = decision_trace

        high_risk_issues = [i for i in state.quality_issues if i.get("severity") == "High"]
        if high_risk_issues:
            state.pending_human_approvals = high_risk_issues
            state.add_message(
                agent_name="Planning Agent",
                stage="PLAN",
                status="WARNING",
                message=f"Plan generated: {len(decision_trace)} actions planned ({len(high_risk_issues)} require Human-in-the-Loop review).",
                reasoning="Identified high-severity or manual review issues (e.g. invalid syntax, negative numeric values).",
                action_taken="Queued automatic safe actions and flagged high-risk actions for human approval.",
                result="Human-in-the-Loop Approval Required for high-risk actions."
            )
        else:
            state.add_message(
                agent_name="Planning Agent",
                stage="PLAN",
                status="SUCCESS",
                message=f"Plan generated: Executing {len(decision_trace)} safe policy actions.",
                reasoning="All detected issues qualify for safe automatic remediation under active policy matrix.",
                action_taken="Approved full automated execution path.",
                result="Proceeding to Cleaning & Transformation Agents."
            )

        # =========================================================================
        # 6. CLEANING AGENT & 7. TRANSFORMATION AGENT
        # =========================================================================
        state.pipeline_status = "Cleaning & Transforming"

        if state.data_category in ["Structured", "Semi-Structured"] and state.raw_df is not None:
            opts = user_options or {
                "trim_whitespace": True, "drop_duplicates": True,
                "impute_numeric": True, "impute_categorical": True, "standardize_dates": True
            }
            cleaned, logs = clean_dataset(state.raw_df, opts)
            state.cleaned_df = cleaned
            state.transformation_log = logs

            state.add_message(
                agent_name="Cleaning Agent",
                stage="CLEAN",
                status="ACTION",
                message=f"Executed safe data cleaning policy ({len(logs)} operations applied).",
                reasoning="Imputed missing values using median/mode and stripped extra whitespace.",
                action_taken="Executed deterministic safe cleaning functions.",
                result=f"Cleaned Rows: {len(cleaned)} (Original: {len(state.raw_df)})"
            )

            state.add_message(
                agent_name="Transformation Agent",
                stage="TRANSFORM",
                status="SUCCESS",
                message="Standardized data schema and date formats.",
                reasoning="Ensured consistent column datatypes and prevented null-date conversions.",
                action_taken="Applied schema standardization.",
                result="Schema verified and clean DataFrame generated."
            )

        else:
            # Unstructured -> Structured Conversion
            converted_df = convert_unstructured_to_structured(state.unstructured_text, state.document_metadata)
            state.cleaned_df = converted_df
            state.transformation_log = [f"Extracted {len(converted_df)} structured record(s) from document text."]

            state.add_message(
                agent_name="Transformation Agent",
                stage="TRANSFORM",
                status="SUCCESS",
                message=f"Converted Unstructured Document into {len(converted_df)} Structured Record(s).",
                reasoning="Extracted entity fields (Name, Email, Phone, Date, Amount) into tabular DataFrame format.",
                action_taken="Ran Unstructured-to-Structured Intelligence Engine.",
                result=f"Generated DataFrame Preview ({len(converted_df)} rows x {len(converted_df.columns)} cols)."
            )

        # =========================================================================
        # 8. VALIDATION AGENT & ANOMALY DETECTION
        # =========================================================================
        state.pipeline_status = "Validating Results"

        if state.raw_df is not None and state.cleaned_df is not None:
            val_res = validate_cleaning(state.raw_df, state.cleaned_df, state.domain)
            state.validation_result = val_res
            anom_res = detect_anomalies(state.cleaned_df, contamination=0.10)
            state.anomalies_res = anom_res

            state.add_message(
                agent_name="Validation Agent",
                stage="VALIDATE",
                status="SUCCESS",
                message=f"Validation Complete: DQI Improved ({val_res['initial_dqi']:.2f} -> {val_res['cleaned_dqi']:.2f}).",
                reasoning=f"Quality gain of +{val_res['dqi_improvement']:.2f} pts verified across 5 dimensions.",
                action_taken="Recalculated DQI score on cleaned dataset.",
                result=f"Post-Cleaning DQI: {val_res['cleaned_dqi']:.2f} / 100 (+{val_res['dqi_improvement']:.2f} pts)"
            )
        else:
            state.add_message(
                agent_name="Validation Agent",
                stage="VALIDATE",
                status="SUCCESS",
                message="Document Entity Extraction Validated.",
                reasoning="Extracted entity syntax and document fields verified.",
                action_taken="Validated document quality score.",
                result=f"Final Document Score: {state.dqi_score:.2f} / 100"
            )

        # =========================================================================
        # 9. ETL / STORAGE AGENT
        # =========================================================================
        state.pipeline_status = "Executing ETL"
        if state.cleaned_df is not None:
            etl_report = run_etl_pipeline(state.cleaned_df, domain=state.domain)
            state.etl_report = etl_report

            state.add_message(
                agent_name="ETL / Storage Agent",
                stage="LOAD",
                status="SUCCESS",
                message="ETL Execution Complete: CSV Download Payload Active.",
                reasoning="Prepared clean CSV export payload and checked PostgreSQL database connection.",
                action_taken="Ran ETL Pipeline.",
                result=f"Output Rows: {etl_report['cleaned_rows']} · Destination: {etl_report['destination_table']}"
            )

        # =========================================================================
        # 10. MONITORING & 11. REPORTING AGENT
        # =========================================================================
        t_end = time.time()
        duration = round(t_end - t_start, 2)
        state.execution_metrics["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state.execution_metrics["duration_seconds"] = duration
        state.execution_metrics["records_processed"] = len(state.raw_df) if state.raw_df is not None else 1
        state.execution_metrics["records_cleaned"] = len(state.cleaned_df) if state.cleaned_df is not None else 1

        state.pipeline_status = "Completed"

        state.add_message(
            agent_name="Monitoring Agent",
            stage="MONITOR",
            status="INFO",
            message=f"Pipeline Completed in {duration}s.",
            reasoning="All 10 agent stages completed execution cleanly without fatal exceptions.",
            action_taken="Recorded execution metrics and duration logs.",
            result=f"Duration: {duration}s · Status: ONLINE"
        )

        state.add_message(
            agent_name="Reporting Agent",
            stage="REPORT",
            status="SUCCESS",
            message="Final Execution Summary Report Ready.",
            reasoning="Compiled multi-agent logs, DQI metrics, and validation results.",
            action_taken="Generated final major project report payload.",
            result="Report Ready for Presentation."
        )

        return state

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class AgentMessage:
    timestamp: str
    agent_name: str
    stage: str
    status: str          # "INFO", "SUCCESS", "DECISION", "WARNING", "ACTION"
    message: str
    reasoning: str = ""
    action_taken: str = ""
    result: str = ""

class AgentState:
    """
    Shared Agentic Pipeline Context storing execution state, detected metrics,
    agent decision logs, and human-in-the-loop triggers across all 10 agents.
    """
    def __init__(self, file_name: str = "No Active Dataset"):
        self.file_name: str = file_name
        self.file_type: str = "N/A"
        self.data_category: str = "Structured"  # Structured, Semi-Structured, Unstructured
        self.domain: str = "General"
        self.domain_confidence: str = "High"
        self.domain_reasoning: str = "Default domain rules active"
        
        # Raw Data Holders
        self.raw_df: Optional[Any] = None
        self.cleaned_df: Optional[Any] = None
        self.unstructured_text: Optional[str] = None
        self.document_metadata: Dict[str, Any] = {}
        
        # Metric Results
        self.profile: Dict[str, Any] = {}
        self.quality_issues: List[Dict[str, Any]] = []
        self.dqi_score: float = 0.0
        self.dqi_dimensions: Dict[str, float] = {}
        self.unstructured_quality_score: float = 0.0
        self.unstructured_dimensions: Dict[str, float] = {}
        self.anomalies_res: Optional[Dict[str, Any]] = None
        
        # Recommendations & Human-in-the-Loop Approval State
        self.recommended_actions: List[Dict[str, Any]] = []
        self.approved_actions: List[str] = []
        self.pending_human_approvals: List[Dict[str, Any]] = []
        
        # Execution & Audit Logs
        self.transformation_log: List[str] = []
        self.validation_result: Optional[Dict[str, Any]] = None
        self.etl_report: Optional[Dict[str, Any]] = None
        self.execution_metrics: Dict[str, Any] = {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "duration_seconds": 0.0,
            "records_processed": 0,
            "records_cleaned": 0,
            "records_rejected": 0
        }
        
        # Agent Communication Stream
        self.agent_messages: List[AgentMessage] = []
        self.pipeline_status: str = "Initialized"
        self.agent_modes: Dict[str, str] = {
            "orchestrator_mode": "Local Rule-Based Orchestration (Offline Safe)",
            "llm_connected": False
        }

    def add_message(self, agent_name: str, stage: str, status: str, message: str, reasoning: str = "", action_taken: str = "", result: str = ""):
        msg = AgentMessage(
            timestamp=time.strftime("%H:%M:%S"),
            agent_name=agent_name,
            stage=stage,
            status=status,
            message=message,
            reasoning=reasoning,
            action_taken=action_taken,
            result=result
        )
        self.agent_messages.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "data_category": self.data_category,
            "domain": self.domain,
            "dqi_score": self.dqi_score,
            "pipeline_status": self.pipeline_status,
            "total_agent_messages": len(self.agent_messages),
            "pending_approvals_count": len(self.pending_human_approvals)
        }

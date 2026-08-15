import json
import os
from core.schemas import VulnerabilityReport
from core.database import write_findings, read_findings, record_calibration
from core.context import current_run_context

MAX_READ_SIZE = 1024 * 1024  # 1 MiB

def read_file(filepath: str) -> str:
    """Reads the content of a file within the allowed jail directory."""
    ctx = current_run_context.get()
    if ctx is None:
        return "Error: No active execution context."
    jail = os.path.realpath(ctx.jail_dir)
    base_dir = os.path.dirname(jail) if os.path.isfile(jail) else jail
    if not os.path.isabs(filepath):
        target_path_real = os.path.realpath(os.path.join(base_dir, filepath))
    else:
        target_path_real = os.path.realpath(filepath)
    
    try:
        if os.path.isfile(jail):
            if target_path_real != jail:
                return f"Error: Permission denied. The filepath {filepath} is outside the allowed scope."
        elif os.path.commonpath([jail, target_path_real]) != jail:
            return f"Error: Permission denied. The filepath {filepath} is outside the allowed directory."
    except ValueError:
        return "Error: Permission denied. Directory outside scope."
        
    if not os.path.isfile(target_path_real):
        return f"Error: File not found at {filepath}"
    try:
        with open(target_path_real, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(MAX_READ_SIZE + 1)
            if len(content) > MAX_READ_SIZE:
                return content[:MAX_READ_SIZE] + f"\n\n[TRUNCATED: File exceeds {MAX_READ_SIZE} characters/bytes limit]"
            return content
    except Exception as e:
        return f"Error reading file: {e}"

def report_findings(report: VulnerabilityReport) -> str:
    """Submit the structured report of all vulnerabilities found in the file."""
    ctx = current_run_context.get()
    if ctx is None:
        return "Error: No active execution context."
    try:
        if isinstance(report, VulnerabilityReport):
            findings = report.findings
        elif isinstance(report, dict):
            report_obj = VulnerabilityReport.model_validate(report)
            findings = report_obj.findings
        elif isinstance(report, list):
            report_obj = VulnerabilityReport(findings=report)
            findings = report_obj.findings
        else:
            return f"ERROR SAVING DB: Invalid report payload type '{type(report).__name__}'."
            
        write_findings(ctx.db_path, ctx.target_file, findings, run_id=ctx.run_id)
        return f"SUCCESS: Saved {len(findings)} finding(s) to database."
    except Exception as e:
        return f"ERROR SAVING DB: {e}"

def get_findings() -> str:
    """Retrieves all recorded vulnerability findings for the current target file."""
    ctx = current_run_context.get()
    if ctx is None:
        return "Error: No active execution context."
    findings = read_findings(ctx.db_path, ctx.target_file, run_id=ctx.run_id)
    if not findings:
        return "No findings recorded for this file."
    return json.dumps(findings, indent=2)

def score_risk(score: int, reasoning: str) -> str:
    """Records the final risk calibration score."""
    ctx = current_run_context.get()
    if ctx is None or not ctx.db_path:
        return "Error: No active execution context or database path."
    if isinstance(score, bool) or not isinstance(score, int) or not (0 <= score <= 100):
        return f"Error: Risk score must be an integer between 0 and 100, got {score!r}."
    try:
        record_calibration(ctx.db_path, ctx.target_file, score, reasoning, run_id=ctx.run_id)
        return f"SUCCESS: Recorded risk score {score}/100. Reasoning: {reasoning}"
    except Exception as e:
        return f"ERROR SAVING RISK SCORE: {e}"

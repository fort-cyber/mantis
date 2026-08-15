from .research_tools import read_file, report_findings, get_findings, score_risk
from .sandbox_tools import run_sandbox, apply_patch

TOOLS: dict[str, object] = {
    "read_file": read_file,
    "report_findings": report_findings,
    "get_findings": get_findings,
    "score_risk": score_risk,
    "run_sandbox": run_sandbox,
    "apply_patch": apply_patch,
}

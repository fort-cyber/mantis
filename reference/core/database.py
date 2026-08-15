import sqlite3
import json
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

@contextmanager
def _db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db(db_path: str):
    """Initialize the SQLite database with findings and risk_scores tables and unique indexes."""
    with _db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                filepath TEXT,
                title TEXT,
                severity TEXT,
                description TEXT,
                line_numbers TEXT NOT NULL DEFAULT '[]',
                remediation TEXT,
                status TEXT NOT NULL DEFAULT 'reported',
                UNIQUE(filepath, title, description, line_numbers, run_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                filepath TEXT,
                score INTEGER,
                reasoning TEXT,
                UNIQUE(filepath, run_id)
            )
        """)

def write_findings(db_path: str, filepath: str, findings: list, run_id: str = "", status: str = "reported"):
    """Write structured findings to the database contextually associated with real path."""
    with _db(db_path) as conn:
        cursor = conn.cursor()
        for obj in findings:
            finding = obj.model_dump() if hasattr(obj, "model_dump") else (obj if isinstance(obj, dict) else dict(obj))
            raw_lines = finding.get("line_numbers")
            if raw_lines and isinstance(raw_lines, (list, tuple, set)):
                try:
                    line_numbers = json.dumps(sorted(list(raw_lines)))
                except Exception:
                    line_numbers = json.dumps(list(raw_lines))
            else:
                line_numbers = "[]"

            finding_status = finding.get("status") or status

            cursor.execute("""
                INSERT OR REPLACE INTO findings (run_id, filepath, title, severity, description, line_numbers, remediation, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                filepath,
                finding.get("title"),
                finding.get("severity"),
                finding.get("description"),
                line_numbers,
                finding.get("remediation"),
                finding_status,
            ))

def update_status(db_path: str, filepath: str, run_id: str, status: str):
    """Update status for all findings of a target file in a given run."""
    with _db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE findings
            SET status = ?
            WHERE filepath = ? AND run_id = ?
        """, (status, filepath, run_id))


def read_findings(
    db_path: str,
    filepath: Optional[str] = None,
    run_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read findings from the database, optionally filtered by filepath, run_id, or status."""
    with _db(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM findings WHERE 1=1"
        params = []
        if filepath:
            query += " AND filepath = ?"
            params.append(filepath)
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY id ASC"
        cursor.execute(query, params)
        rows = []
        for r in cursor.fetchall():
            row_dict = dict(r)
            if row_dict.get("line_numbers"):
                try:
                    parsed = json.loads(row_dict["line_numbers"])
                    row_dict["line_numbers"] = parsed if parsed else None
                except Exception:
                    row_dict["line_numbers"] = None
            else:
                row_dict["line_numbers"] = None
            rows.append(row_dict)
        return rows

def record_calibration(db_path: str, filepath: str, score: int, reasoning: str, run_id: str = ""):
    """Record final risk calibration score into the database with idempotent overwrite on retry."""
    with _db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO risk_scores (run_id, filepath, score, reasoning)
            VALUES (?, ?, ?, ?)
        """, (run_id, filepath, score, reasoning))

def read_risk_scores(db_path: str, filepath: Optional[str] = None, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read risk scores from the database, optionally filtered by filepath and run_id."""
    with _db(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM risk_scores WHERE 1=1"
        params = []
        if filepath:
            query += " AND filepath = ?"
            params.append(filepath)
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        query += " ORDER BY id ASC"
        cursor.execute(query, params)
        return [dict(r) for r in cursor.fetchall()]


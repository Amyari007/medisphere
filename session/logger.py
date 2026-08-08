"""
Persists real session data to CSV and SQLite so a therapist could compare
sessions over time. Every write here comes from actual measured session
data passed in by the caller — this module doesn't invent any values.
"""

import csv
import os
import sqlite3
from datetime import datetime, timezone


REP_FIELDS = ["session_id", "rep_index", "timestamp", "duration_s", "peak_openness", "peak_velocity_px_s"]


class SessionLogger:
    def __init__(self, db_path="session_data/medisphere.db", csv_dir="session_data"):
        self.db_path = db_path
        self.csv_dir = csv_dir
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reps (
                session_id TEXT,
                rep_index INTEGER,
                timestamp TEXT,
                duration_s REAL,
                peak_openness REAL,
                peak_velocity_px_s REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_summary (
                session_id TEXT PRIMARY KEY,
                started_at TEXT,
                total_reps INTEGER,
                motor_score REAL,
                fatigue_declining INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def log_rep(self, rep_index, rep_data):
        row = {
            "session_id": self.session_id,
            "rep_index": rep_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": rep_data.get("duration_s"),
            "peak_openness": rep_data.get("peak_openness"),
            "peak_velocity_px_s": rep_data.get("peak_velocity_px_s"),
        }
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO reps (session_id, rep_index, timestamp, duration_s, peak_openness, peak_velocity_px_s) "
            "VALUES (:session_id, :rep_index, :timestamp, :duration_s, :peak_openness, :peak_velocity_px_s)",
            row,
        )
        conn.commit()
        conn.close()

        csv_path = os.path.join(self.csv_dir, f"{self.session_id}_reps.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REP_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def finalize_session(self, total_reps, motor_score, fatigue_declining):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO session_summary "
            "(session_id, started_at, total_reps, motor_score, fatigue_declining) VALUES (?, ?, ?, ?, ?)",
            (self.session_id, self.session_id, total_reps, motor_score, int(bool(fatigue_declining))),
        )
        conn.commit()
        conn.close()

    def read_all_reps(self):
        """For verification/testing: reads back everything logged this session."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM reps WHERE session_id = ? ORDER BY rep_index", (self.session_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def generate_report(self):
        """Plain-text progress report for the current session, from real logged data."""
        reps = self.read_all_reps()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        summary = conn.execute(
            "SELECT * FROM session_summary WHERE session_id = ?", (self.session_id,)
        ).fetchone()
        conn.close()

        lines = [f"MediSphere Session Report — {self.session_id}", "-" * 40]
        lines.append(f"Total repetitions: {len(reps)}")
        if reps:
            durations = [r["duration_s"] for r in reps if r["duration_s"] is not None]
            openness = [r["peak_openness"] for r in reps if r["peak_openness"] is not None]
            if durations:
                lines.append(f"Avg rep duration: {sum(durations) / len(durations):.2f}s")
            if openness:
                lines.append(f"Avg peak openness: {sum(openness) / len(openness):.2f}")
        if summary:
            lines.append(f"Motor score: {summary['motor_score']}")
            lines.append(f"Fatigue trend detected: {bool(summary['fatigue_declining'])}")
        report_text = "\n".join(lines)

        report_path = os.path.join(self.csv_dir, f"{self.session_id}_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)
        return report_path, report_text

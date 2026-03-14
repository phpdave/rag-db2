"""
ibmi.py — IBM i (AS/400) connectivity via pyodbc + IBM i Access ODBC Driver.

NOTE: This connects to IBM i (DB2 for i), which is a completely different
platform from IBM DB2 LUW or DB2 z/OS. Do NOT use ibm_db here — that is the
driver for LUW/z/OS. IBM i requires the IBM i Access ODBC Driver (ibm-iaccess).

Driver package: ibm-iaccess (apt)
ODBC driver name: IBM i Access ODBC Driver 64-bit
Connection key: SYSTEM=<hostname>  (not HOSTNAME, not SERVER)
"""

import os
import re

import pyodbc


def _conn_str() -> str:
    host           = os.environ.get("IBMI_HOST", "pub400.com")
    user           = os.environ["IBMI_USER"]
    password       = os.environ["IBMI_PASSWORD"]
    default_schema = os.environ.get("IBMI_DEFAULT_SCHEMA", "")
    conn = (
        "DRIVER={IBM i Access ODBC Driver 64-bit};"
        f"SYSTEM={host};"
        f"UID={user};"
        f"PWD={password};"
        "TRANSLATE=1;"
        "UNICODESQL=1;"
    )
    if default_schema:
        conn += f"DefaultLibraries={default_schema};"
    return conn


def get_conn() -> pyodbc.Connection:
    return pyodbc.connect(_conn_str(), autocommit=True)


def extract_sql(text: str) -> str | None:
    """Return the first SELECT/WITH statement found in a markdown ```sql block."""
    blocks = re.findall(r'```(?:sql)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    for block in blocks:
        block = block.strip()
        if re.match(r'\s*(SELECT|WITH)\b', block, re.IGNORECASE):
            return block
    return None


def extract_all_sql(text: str) -> str | None:
    """Return the first ```sql block found, regardless of statement type."""
    blocks = re.findall(r'```(?:sql)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    for block in blocks:
        block = block.strip()
        if block:
            return block
    return None


def _split_statements(sql: str) -> list[str]:
    """Split a SQL block on semicolons, stripping comments and blank statements."""
    # Remove single-line comments
    sql = re.sub(r'--[^\n]*', '', sql)
    parts = [s.strip() for s in sql.split(';')]
    return [p for p in parts if p]


def run_all_statements(sql: str) -> list[dict]:
    """
    Execute every statement in a SQL block (DDL, DML, SELECT) sequentially.
    Returns a list of result dicts, one per statement.
    """
    statements = _split_statements(sql)
    results = []
    for stmt in statements:
        results.append(run_statement(stmt))
    return results


def _add_row_limit(sql: str, limit: int = 100) -> str:
    """Append FETCH FIRST N ROWS ONLY if not already present."""
    if not re.search(r'FETCH\s+FIRST', sql, re.IGNORECASE):
        sql = sql.rstrip('; \n') + f"\nFETCH FIRST {limit} ROWS ONLY"
    return sql


def run_sql(sql: str) -> dict:
    """
    Execute a SELECT/WITH statement against IBM i and return results as a dict.
    Only SELECT / WITH statements are permitted.
    """
    if not re.match(r'\s*(SELECT|WITH)\b', sql, re.IGNORECASE):
        return {"success": False, "error": "Only SELECT/WITH statements are permitted."}

    sql = _add_row_limit(sql)

    try:
        conn   = get_conn()
        cursor = conn.cursor()
        cursor.execute(sql)

        columns = [col[0] for col in cursor.description]
        rows    = [
            [str(v) if v is not None else None for v in row]
            for row in cursor.fetchall()
        ]
        cursor.close()
        conn.close()

        return {
            "success":   True,
            "sql":       sql,
            "columns":   columns,
            "rows":      rows,
            "row_count": len(rows),
        }
    except Exception as exc:
        return {"success": False, "sql": sql, "error": str(exc)}


def run_statement(sql: str) -> dict:
    """
    Execute any SQL statement (DDL or DML) against IBM i.
    Returns rows for SELECT/WITH; row_count affected for INSERT/UPDATE/DELETE/CREATE/DROP.
    """
    sql = sql.strip().rstrip(";")
    is_query = bool(re.match(r'\s*(SELECT|WITH)\b', sql, re.IGNORECASE))

    if is_query:
        sql = _add_row_limit(sql)

    try:
        conn   = get_conn()
        cursor = conn.cursor()
        cursor.execute(sql)

        if is_query and cursor.description:
            columns = [col[0] for col in cursor.description]
            rows    = [
                [str(v) if v is not None else None for v in row]
                for row in cursor.fetchall()
            ]
            cursor.close()
            conn.close()
            return {"success": True, "sql": sql, "columns": columns, "rows": rows, "row_count": len(rows)}
        else:
            affected = cursor.rowcount
            cursor.close()
            conn.close()
            return {"success": True, "sql": sql, "rows_affected": affected}

    except Exception as exc:
        return {"success": False, "sql": sql, "error": str(exc)}

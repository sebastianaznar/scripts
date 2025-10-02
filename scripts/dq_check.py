import sys, argparse, logging
from pathlib import Path
import pandas as pd, yaml
 
# --- Rutas robustas relativas a este archivo ---
HERE = Path(__file__).resolve().parent     # .../project/scripts
PROJECT_ROOT = HERE.parent                 # .../project
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
OUT_DIR = PROJECT_ROOT / "out"
 
# --- Logging y argumentos CLI ---
def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
 
def parse_args():
    p = argparse.ArgumentParser(description="Data Quality Gate (simple)")
    p.add_argument("-i","--input", required=True, help="Ruta al CSV de entrada")
    p.add_argument("-r","--rules", default=str(CONFIG_DIR/"rules.yml"),
                   help="Ruta al archivo de reglas YAML")
    p.add_argument("-o","--out", default=str(OUT_DIR),
                   help="Carpeta para el informe")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()
 
# --- Reglas (YAML) ---
def load_rules(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
 
# --- Renombrado y tipos ---
def rename_columns_to_canonical(df: pd.DataFrame, columns_map: dict) -> pd.DataFrame:
    if not columns_map:
        return df
    real_to_canon = {v: k for k, v in columns_map.items()}
    return df.rename(columns=lambda c: real_to_canon.get(c, c))
 
def coerce_types(df: pd.DataFrame, types_cfg: dict) -> pd.DataFrame:
    if not types_cfg:
        return df.copy()
    out = df.copy()
    for col, t in types_cfg.items():
        if col not in out.columns:  # si la columna no existe, seguimos
            continue
        try:
            if t == "int":
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
            elif t == "float":
                out[col] = pd.to_numeric(out[col], errors="coerce")
            elif t == "datetime":
                out[col] = pd.to_datetime(out[col], errors="coerce", dayfirst=True)
            elif t == "category":
                out[col] = out[col].astype("string").astype("category")
            elif t == "string":
                out[col] = out[col].astype("string")
        except Exception as e:
            logging.warning("No se pudo convertir %s a %s: %s", col, t, e)
    return out
 
# --- Checks: devuelven lista de 'findings' dicts ---
def check_schema(df, rules):
    required = set((rules.get("schema") or {}).get("required", []))
    missing = sorted(list(required - set(df.columns)))
    findings = []
    for col in missing:
        findings.append({"rule":"missing_column","column":col,"row":None,
                         "detail":"Columna obligatoria ausente"})
    return findings
 
def check_domains(df, rules):
    findings = []
    for col, cfg in (rules.get("domains") or {}).items():
        if col not in df.columns:
            continue
        allowed = set(cfg.get("allowed", []))
        mask = (~df[col].isna()) & (~df[col].isin(allowed))
        for idx in df[mask].index:
            findings.append({"rule":"domain_violation","column":col,"row":int(idx),
                             "detail":f"Valor={df.at[idx,col]!r} no permitido"})
    return findings
 
def check_ranges(df, rules):
    findings = []
    for col, cfg in (rules.get("ranges") or {}).items():
        if col not in df.columns:
            continue
        mn, mx = cfg.get("min"), cfg.get("max")
        if mn is not None:
            bad = (df[col] < mn)
            for idx in df[bad.fillna(False)].index:
                findings.append({"rule":"range_violation","column":col,"row":int(idx),
                                 "detail":f"< min({mn})"})
        if mx is not None:
            bad = (df[col] > mx)
            for idx in df[bad.fillna(False)].index:
                findings.append({"rule":"range_violation","column":col,"row":int(idx),
                                 "detail":f"> max({mx})"})
    return findings
 
def check_uniques(df, rules):
    findings = []
    for cols in (rules.get("uniques") or []):
        dups = df.duplicated(subset=cols, keep=False)
        for idx in df[dups].index:
            findings.append({"rule":"uniqueness_violation",
                             "column":",".join(cols),"row":int(idx),
                             "detail":"Fila duplicada para la clave dada"})
    return findings
 
def check_conditionals(df, rules):
    findings = []
    for cond in (rules.get("conditionals") or []):
        if "rule" in cond:
            expr = cond["rule"]                     # p.ej. "n_adultos + n_menores >= 1"
            bad = ~df.eval(expr)                   # incumplimientos
            for idx in df[bad.fillna(True)].index:
                findings.append({"rule":"conditional_violation","column":None,
                                 "row":int(idx),"detail":f"No cumple: {expr}"})
        else:
            if_ = cond.get("if", {})
            then = cond.get("then", {})
            mask = pd.Series(True, index=df.index)
            for c, v in if_.items():
                mask = mask & (df[c] == v)
            req = then.get("require_not_null", []) or []
            for col in req:
                bad = mask & df[col].isna()
                for idx in df[bad].index:
                    findings.append({"rule":"conditional_violation","column":col,
                                     "row":int(idx),"detail":f"{col} requerido cuando {if_}"})
    return findings
 
def apply_severity(findings, rules):
    sev_map = (rules.get("severity") or {})
    for f in findings:
        f["severity"] = sev_map.get(f["rule"], "warning")
 
# --- Programa principal ---
def main():
    args = parse_args()
    setup_logging(args.log_level)
 
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    logging.info("Leyendo CSV: %s", args.input)
    df_raw = pd.read_csv(args.input)  # ajusta sep/encoding si tu CSV lo requiere
    rules = load_rules(Path(args.rules))
 
    df = rename_columns_to_canonical(df_raw, rules.get("columns_map"))
    df = coerce_types(df, rules.get("types"))
 
    findings = []
    findings += check_schema(df, rules)
    findings += check_domains(df, rules)
    findings += check_ranges(df, rules)
    findings += check_uniques(df, rules)
    findings += check_conditionals(df, rules)
 
    apply_severity(findings, rules)
    summary = {}
    for f in findings:
        summary[f["severity"]] = summary.get(f["severity"], 0) + 1
    logging.info("Incidencias por severidad: %s", summary or "sin incidencias")
 
    report_path = out_dir / "dq_report.csv"
    pd.DataFrame(findings).to_csv(report_path, index=False)
    logging.info("Informe: %s", report_path)
 
    has_errors = any(f["severity"] == "error" for f in findings)
    if has_errors:
        logging.error("Errores críticos detectados. Abortando (exit=1).")
        sys.exit(1)
    else:
        logging.info("Calidad OK. (exit=0)")
        sys.exit(0)
 
if __name__ == "__main__":
    main()
import json
import xlsxwriter
from pathlib import Path

def json_to_xlsx(json_path: str, xlsx_path: str = None, sheet_name: str = "debugData"):
    """
    Convert a debugData-style JSON file into an Excel .xlsx file using xlsxwriter.

    - If entries are a dict keyed by frame/index, the key becomes 'id'.
    - If entries are a list, indices (1-based) become 'id'.
    - Supports a 'section' field:
        * If present per-record, it's written for that record.
        * If present once at top-level, it's copied to every record.
    - 'loc': [x, y] is split into 'loc_x' and 'loc_y'.
    - All other fields are written as columns.
    """
    in_path = Path(json_path)
    if xlsx_path is None:
        xlsx_path = str(in_path.with_suffix(".xlsx"))

    # Load JSON
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract a global/top-level section if present (applies to all rows)
    global_meta = {}
    if isinstance(data, dict) and "section" in data and not isinstance(data.get("section"), dict):
        global_meta["section"] = data["section"]

    # Helper to make values Excel-friendly (numbers stay numbers; others -> JSON strings)
    def normalize_cell_value(v):
        if isinstance(v, (int, float)) or v is None or isinstance(v, str):
            return v if v is not None else ""
        # lists, dicts, tuples, etc. -> JSON string
        try:
            return json.dumps(v)
        except Exception:
            return str(v)

    # Normalize into a list of records
    records = []
    if isinstance(data, dict):
        # Treat only dict-valued items as records; skip scalar top-level meta like 'section'
        items = [(k, v) for k, v in data.items() if isinstance(v, dict)]
        # If there were no dict-valued items, fall back to iterating everything (original behavior)
        if not items:
            items = list(data.items())
    elif isinstance(data, list):
        items = list(enumerate(data, start=1))
    else:
        raise ValueError("Unsupported JSON top-level type (expected dict or list).")

    for key, val in items:
        rec = {"id": str(key)}
        # Carry global/top-level meta (e.g., section) onto each row
        rec.update(global_meta)

        if isinstance(val, dict):
            # Split loc if present
            loc = val.get("loc")
            if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                rec["loc_x"] = loc[0]
                rec["loc_y"] = loc[1]

            # Add the rest of the fields (including per-record 'section' if present)
            for k, v in val.items():
                if k == "loc":
                    continue
                rec[k] = normalize_cell_value(v)
        else:
            # Not a dict; store under a generic column
            rec["value"] = normalize_cell_value(val)

        # Normalize cell values for any leftover raw types
        for k in list(rec.keys()):
            if k in ("loc_x", "loc_y"):  # already numeric-friendly
                continue
            rec[k] = normalize_cell_value(rec[k])

        records.append(rec)

    # Collect all headers seen across records
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())

    # Prefer a stable, friendly header order (include 'section')
    preferred = ["id", "section", "loc_x", "loc_y", "throttle", "brake", "steer", "speed", "lap"]
    remaining = [k for k in sorted(all_keys) if k not in preferred]
    headers = [k for k in preferred if k in all_keys] + remaining

    # Write Excel
    workbook = xlsxwriter.Workbook(xlsx_path)
    ws = workbook.add_worksheet(sheet_name)

    header_fmt = workbook.add_format({"bold": True, "bg_color": "#EEEEEE", "border": 1})
    num_fmt = workbook.add_format({"num_format": "0.000"})
    general_fmt = workbook.add_format()

    # Headers
    for col, h in enumerate(headers):
        ws.write(0, col, h, header_fmt)

    # Rows
    for row_idx, r in enumerate(records, start=1):
        for col, h in enumerate(headers):
            val = r.get(h, "")
            if isinstance(val, (int, float)):
                ws.write_number(row_idx, col, val, num_fmt)
            else:
                ws.write(row_idx, col, "" if val is None else val, general_fmt)

    # Light autofit
    for col in range(len(headers)):
        ws.set_column(col, col, 14)

    workbook.close()
    return xlsx_path

if __name__ == "__main__":
    # Change these if you run from a different working directory
    input_json = r".\debugData\debugData.json"
    output_xlsx = r".\debugData\debugData.xlsx"
    print("Wrote:", json_to_xlsx(input_json, output_xlsx))
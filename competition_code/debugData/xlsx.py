import json
import xlsxwriter
from pathlib import Path

def json_to_xlsx(json_path: str, xlsx_path: str = None, sheet_name: str = "debugData"):
    """
    Convert a debugData-style JSON file into an Excel .xlsx file using xlsxwriter.

    - If entries are a dict keyed by frame/index, the key becomes 'id'.
    - If entries are a list, indices (1-based) become 'id'.
    - 'loc': [x, y] is split into 'loc_x' and 'loc_y'.
    - All other fields are written as columns.
    """
    in_path = Path(json_path)
    if xlsx_path is None:
        xlsx_path = str(in_path.with_suffix(".xlsx"))

    # Load JSON
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize into a list of records
    records = []
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = enumerate(data, start=1)
    else:
        raise ValueError("Unsupported JSON top-level type (expected dict or list).")

    for key, val in items:
        rec = {"id": str(key)}
        if isinstance(val, dict):
            loc = val.get("loc")
            if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                rec["loc_x"] = loc[0]
                rec["loc_y"] = loc[1]
            # Add the rest of the fields
            for k, v in val.items():
                if k != "loc":
                    rec[k] = v
        else:
            # Not a dict; store under a generic column
            rec["value"] = val
        records.append(rec)

    # Collect all headers seen across records
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())

    # Prefer a stable, friendly header order
    preferred = ["id", "loc_x", "loc_y", "throttle", "brake", "steer", "speed", "lap"]
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
                ws.write(row_idx, col, val, general_fmt)

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

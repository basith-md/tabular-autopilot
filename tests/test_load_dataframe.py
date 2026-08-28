import io

import pandas as pd

from tabular_autopilot.pipeline import load_dataframe

CSV_HEADER = "name,rating\n"

# Raw byte 0x92 -- cp1252's right single quotation mark ('’'), and
# exactly what a real "exported from Excel on Windows" CSV produces for an
# apostrophe. Built from actual bytes (not a "\x92" string escape, which is
# Unicode U+0092 -- a different, cp1252-unencodable codepoint).
_CP1252_ROW = CSV_HEADER.encode("ascii") + b"Sam\x92s Place,4\n"


def test_loads_plain_utf8_csv(tmp_path):
    path = tmp_path / "plain.csv"
    path.write_text(CSV_HEADER + "Café Luna,5\n", encoding="utf-8")

    df = load_dataframe(path)

    assert list(df.columns) == ["name", "rating"]
    assert df.loc[0, "name"] == "Café Luna"


def test_loads_cp1252_csv_with_smart_quote_from_path(tmp_path):
    # Byte 0x92 is cp1252's right single quotation mark -- invalid as UTF-8,
    # and exactly what a real-world "exported from Excel on Windows" CSV
    # produces for a name like "Sam's Place".
    path = tmp_path / "cp1252.csv"
    path.write_bytes(_CP1252_ROW)

    df = load_dataframe(path)

    assert df.loc[0, "name"] == "Sam’s Place"


def test_loads_cp1252_csv_from_file_like_buffer():
    # Streamlit's file_uploader and the browser demo's FS both hand back a
    # buffer, not a path -- the encoding-fallback loop must reset it with
    # seek(0) between attempts rather than reading a partially-consumed stream.
    buf = io.BytesIO(_CP1252_ROW)

    df = load_dataframe(buf, filename="cp1252.csv")

    assert df.loc[0, "name"] == "Sam’s Place"


def test_loads_excel_file(tmp_path):
    path = tmp_path / "sheet.xlsx"
    pd.DataFrame({"name": ["A", "B"], "rating": [1, 2]}).to_excel(path, index=False)

    df = load_dataframe(path)

    assert list(df.columns) == ["name", "rating"]
    assert len(df) == 2


def test_loads_excel_from_buffer_using_filename_hint():
    buf = io.BytesIO()
    pd.DataFrame({"name": ["A"], "rating": [1]}).to_excel(buf, index=False)
    buf.seek(0)

    df = load_dataframe(buf, filename="uploaded.xlsx")

    assert list(df.columns) == ["name", "rating"]

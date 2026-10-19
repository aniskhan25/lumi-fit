from lumi_fit.tui import _COMPACT_COLUMNS, _WIDE_COLUMNS, _short, _table_line


def _heading(columns):
    return _table_line({column.key: column.heading for column in columns}, columns)


def _sample_row(columns):
    return _table_line(
        {
            "marker": "›",
            "model": "mistralai/Mistral-7B-Instruct-v0.3-with-a-long-suffix",
            "params": "7.2B",
            "eligibility": "ELIGIBLE",
            "hbm": "54%",
            "safe": "25",
            "rate": "136.8",
            "gcds": "4",
            "cost": "2.0",
            "estimated": "100.0",
            "quality": "NOT REQ",
            "fit": "UNSCORED",
        },
        columns,
    )


def test_table_headings_and_rows_share_exact_column_boundaries() -> None:
    for columns in (_COMPACT_COLUMNS, _WIDE_COLUMNS):
        heading = _heading(columns)
        row = _sample_row(columns)
        assert len(heading) == len(row)

        offset = 0
        for column in columns:
            heading_cell = heading[offset : offset + column.width]
            row_cell = row[offset : offset + column.width]
            assert heading_cell.strip() == column.heading
            assert len(row_cell) == column.width
            offset += column.width + 1


def test_long_model_names_keep_the_publisher_visible() -> None:
    assert _short("mistralai/Mistral-7B-Instruct-v0.3", 26) == "mistralai/Mistral-7B-Inst…"

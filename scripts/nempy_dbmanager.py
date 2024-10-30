"""
Script to create and prefill an SQLite database with AEMO MMS data.
"""

import sqlite3
from nempy.historical_inputs.mms_db import DBManager
import argparse
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "p",
        action="store",
        help="AEMO MMS SQLite database name.",
        default="db/historical.db",
    )
    parser.add_argument(
        "-l",
        "--load_data",
        action="store_true",
        help="Store data or not",
        default=False,
    )

    args = parser.parse_args()
    path = Path(Path(__file__).parent.parent, args.p).expanduser()

    con = sqlite3.connect(path)

    historical = DBManager(con)

    if args.load_data:
        historical.populate(start_year=2023, start_month=1, end_year=2024, end_month=5)

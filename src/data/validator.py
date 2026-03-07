import pandas as pd

REQUIRED_COLS = ["SEASON", "BASIN", "ISO_TIME", "LAT", "LON", "USA ATCF_ID"]


def validate_storm_data(df: pd.DataFrame) -> dict:
    """Run data quality checks and return quarantined rows.

    Prints a pass/fail report per check.
    Returns a dict mapping check name -> DataFrame of bad rows.
    """
    quarantine: dict[str, pd.DataFrame] = {}
    width = 30

    def _report(name: str, bad: pd.DataFrame):
        status = "PASS" if len(bad) == 0 else "FAIL"
        print(f"  [{status}] {name:<{width}} bad rows: {len(bad):,}")
        if len(bad):
            quarantine[name] = bad

    print("=" * 55)
    print("Storm Data Validation Report")
    print("=" * 55)

    # 1. Required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"  [FAIL] {'Required columns':<{width}} missing: {missing}")
        quarantine["required_columns"] = pd.DataFrame({"missing_column": missing})
    else:
        print(f"  [PASS] {'Required columns':<{width}}")

    # 2. LAT range [-90, 90]
    if "LAT" in df.columns:
        mask = df["LAT"].notna() & ~df["LAT"].between(-90, 90)
        _report("LAT range [-90, 90]", df[mask])

    # 3. LON range [-180, 180]
    if "LON" in df.columns:
        mask = df["LON"].notna() & ~df["LON"].between(-180, 180)
        _report("LON range [-180, 180]", df[mask])

    # 4. SEASON range (1800–2100)
    if "SEASON" in df.columns:
        mask = df["SEASON"].notna() & ~df["SEASON"].between(1800, 2100)
        _report("SEASON range [1800, 2100]", df[mask])

    # 5. ISO_TIME parseable (NaT after coercion means unparseable)
    if "ISO_TIME" in df.columns:
        mask = df["ISO_TIME"].isna()
        _report("ISO_TIME parseable", df[mask])

    # 6. USA WIND > 0 and < 250 (when not null)
    if "USA WIND" in df.columns:
        mask = df["USA WIND"].notna() & ~df["USA WIND"].between(1, 249)
        _report("USA WIND in (0, 250)", df[mask])

    # 7. USA PRES in [800, 1050] (when not null)
    if "USA PRES" in df.columns:
        mask = df["USA PRES"].notna() & ~df["USA PRES"].between(800, 1050)
        _report("USA PRES in [800, 1050]", df[mask])

    # 8. Duplicates: same USA ATCF_ID + ISO_TIME
    if "USA ATCF_ID" in df.columns and "ISO_TIME" in df.columns:
        dup_mask = df.duplicated(subset=["USA ATCF_ID", "ISO_TIME"], keep=False)
        _report("Duplicate ATCF_ID + ISO_TIME", df[dup_mask])

    # 9. Null core fields: LAT, LON, ISO_TIME
    core = [c for c in ["LAT", "LON", "ISO_TIME"] if c in df.columns]
    if core:
        null_mask = df[core].isna().any(axis=1)
        _report("Null in LAT / LON / ISO_TIME", df[null_mask])

    print("=" * 55)
    total_checks = len(quarantine)
    print(f"Checks with issues: {total_checks}")
    print("=" * 55)

    return quarantine

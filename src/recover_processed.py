"""Rebuild missing train/validation feature tables without changing other artifacts.

Run from the project directory with: python -m src.recover_processed
"""
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C
from .features import apply_preprocess, build_features, clean


CHUNK_ROWS = 50_000
PATIENTS_PER_FEATURE_BATCH = 500
META_COLUMNS = ["patient_id", "hour", "label"]


def _id_mask(values: pd.Series, allowed_ids: set[str]) -> np.ndarray:
    return np.fromiter(
        (value in allowed_ids for value in values.to_numpy(dtype=object)),
        dtype=bool,
        count=len(values),
    )


def _validate_feature_table(path: Path, expected_ids: set[str], expected_columns: list[str]) -> int:
    schema = pq.read_schema(path)
    if schema.names != expected_columns:
        raise ValueError(f"{path.name} columns do not match the saved feature schema.")
    parquet_file = pq.ParquetFile(path)
    try:
        observed_ids: set[str] = set()
        for batch in parquet_file.iter_batches(columns=["patient_id"], batch_size=100_000):
            observed_ids.update(batch.column(0).to_pylist())
        row_count = parquet_file.metadata.num_rows
    finally:
        parquet_file.close()
    if observed_ids != expected_ids:
        raise ValueError(
            f"{path.name} patient IDs mismatch: "
            f"missing={len(expected_ids - observed_ids)}, "
            f"unexpected={len(observed_ids - expected_ids)}"
        )
    return row_count


def _read_ids(name: str) -> set[str]:
    path = C.DATA_SPLITS / f"{name}_ids.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Required split ID file is missing: {path}")
    ids = path.read_text().splitlines()
    result = {patient_id.strip() for patient_id in ids if patient_id.strip()}
    if len(result) != len([patient_id for patient_id in ids if patient_id.strip()]):
        raise ValueError(f"Duplicate patient IDs found in {path}")
    return result


def _output_batch(
    rows: pd.DataFrame,
    pipe: dict,
    features: list[str],
    split_ids: dict[str, set[str]],
    seen_patients: dict[str, set[str]],
    writers: dict[str, pq.ParquetWriter],
    stack: ExitStack,
    temp_paths: dict[str, Path],
) -> dict[str, int]:
    counts = {"train": 0, "val": 0}
    if rows.empty:
        return counts

    rows, _ = clean(rows.copy())
    patient_ids = rows[C.ID].drop_duplicates().tolist()
    expected_columns = META_COLUMNS + features

    for start in range(0, len(patient_ids), PATIENTS_PER_FEATURE_BATCH):
        batch_ids = patient_ids[start:start + PATIENTS_PER_FEATURE_BATCH]
        part = rows[rows[C.ID].isin(batch_ids)]
        meta, X = build_features(part)
        Xp = apply_preprocess(X, pipe)
        output = pd.concat(
            [
                meta[[C.ID, C.TIME, C.LABEL]].rename(
                    columns={C.ID: "patient_id", C.TIME: "hour", C.LABEL: "label"}
                ).reset_index(drop=True),
                Xp.reset_index(drop=True),
            ],
            axis=1,
        )
        output["patient_id"] = output["patient_id"].astype(str)
        output["hour"] = pd.to_numeric(output["hour"], errors="raise").astype("int32")
        output["label"] = pd.to_numeric(output["label"], errors="raise").astype("int8")
        output = output[expected_columns]
        if list(output.columns) != expected_columns:
            raise ValueError("Generated feature columns do not match features.json")

        for split_name in ("train", "val"):
            mask = _id_mask(output["patient_id"], split_ids[split_name])
            if not mask.any():
                continue
            selected = output.loc[mask]
            selected_ids = set(selected["patient_id"].unique())
            seen_patients[split_name].update(selected_ids)
            table = pa.Table.from_pandas(selected, preserve_index=False)
            if split_name not in writers:
                fd, temp_name = tempfile.mkstemp(
                    prefix=f".{split_name}.recovery-",
                    suffix=".parquet.tmp",
                    dir=C.DATA_PROC,
                )
                os.close(fd)
                temp_paths[split_name] = Path(temp_name)
                writer = pq.ParquetWriter(temp_name, table.schema, compression="snappy")
                writers[split_name] = writer
                stack.callback(writer.close)
            writers[split_name].write_table(table)
            counts[split_name] += len(selected)
    return counts


def _process_complete_rows(
    rows: pd.DataFrame,
    completed_ids: set[str],
    split_ids: dict[str, set[str]],
    pipe: dict,
    features: list[str],
    seen_patients: dict[str, set[str]],
    writers: dict[str, pq.ParquetWriter],
    stack: ExitStack,
    temp_paths: dict[str, Path],
) -> dict[str, int]:
    counts = {"train": 0, "val": 0}
    if rows.empty:
        return counts

    ids = rows[C.ID].astype(str)
    run_ids = ids[ids.ne(ids.shift())].tolist()
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("A patient appears in separate row groups; raw rows are not patient-contiguous.")
    repeated = completed_ids.intersection(run_ids)
    if repeated:
        raise ValueError(f"Patient IDs recur after their rows ended: {sorted(repeated)[:5]}")
    completed_ids.update(run_ids)

    wanted_ids = split_ids["train"] | split_ids["val"]
    selected = rows.loc[_id_mask(rows[C.ID], wanted_ids)]
    return _output_batch(
        selected,
        pipe,
        features,
        split_ids,
        seen_patients,
        writers,
        stack,
        temp_paths,
    )


def _write_split_files() -> None:
    targets = {name: C.DATA_PROC / f"{name}.parquet" for name in ("train", "val")}

    test_path = C.DATA_PROC / "test.parquet"
    if not test_path.is_file():
        raise FileNotFoundError(f"Existing test data is required and was not found: {test_path}")
    test_stat = test_path.stat()

    split_ids = {name: _read_ids(name) for name in ("train", "val", "test")}
    if split_ids["train"] & split_ids["val"] or split_ids["train"] & split_ids["test"] \
            or split_ids["val"] & split_ids["test"]:
        raise ValueError("Existing patient split ID files overlap.")
    features_path = C.MODELS / "features.json"
    pipeline_path = C.MODELS / "preprocess.joblib"
    if not features_path.is_file() or not pipeline_path.is_file():
        raise FileNotFoundError("Saved features.json and preprocess.joblib are both required.")

    features = json.loads(features_path.read_text())
    pipe = joblib.load(pipeline_path)
    if features != pipe.get("features"):
        raise ValueError("features.json does not match the saved preprocessing pipeline.")
    if len(features) != len(set(features)):
        raise ValueError("features.json contains duplicate feature names.")

    expected_columns = META_COLUMNS + features
    try:
        test_columns = pq.read_schema(test_path).names
    except pa.ArrowInvalid:
        # config.write_table supports pickle data under the .parquet extension.
        test_table = C.read_table(test_path)
        test_columns = test_table.columns.tolist()
        del test_table
    if test_columns != expected_columns:
        raise ValueError(
            "The existing test.parquet columns do not match metadata plus saved features."
        )

    for raw_name in C.RAW_FILES:
        raw_path = C.DATA_RAW / raw_name
        if not raw_path.is_file():
            raise FileNotFoundError(f"Required raw CSV is missing: {raw_path}")

    temp_paths: dict[str, Path] = {}
    created_temp_paths: set[Path] = set()
    completed_splits: set[str] = set()
    row_counts: dict[str, int] = {}
    for split_name, target_path in targets.items():
        if target_path.exists():
            row_counts[split_name] = _validate_feature_table(
                target_path, split_ids[split_name], expected_columns
            )
            completed_splits.add(split_name)
            print(f"Verified existing {split_name}.parquet; leaving it unchanged.")
            continue

        candidates = list(C.DATA_PROC.glob(f".{split_name}.recovery-*.parquet.tmp"))
        if len(candidates) > 1:
            raise FileExistsError(
                f"Multiple temporary {split_name} files found; inspect them before recovery."
            )
        if candidates:
            temp_path = candidates[0]
            row_counts[split_name] = _validate_feature_table(
                temp_path, split_ids[split_name], expected_columns
            )
            temp_paths[split_name] = temp_path
            completed_splits.add(split_name)
            print(f"Verified staged {split_name} data; it can be finalized safely.")

    active_splits = {"train", "val"} - completed_splits
    process_split_ids = {
        name: split_ids[name] if name in active_splits else set()
        for name in ("train", "val")
    }
    seen_patients = {"train": set(), "val": set()}
    completed_ids: set[str] = set()
    written_rows = {"train": 0, "val": 0}
    writers: dict[str, pq.ParquetWriter] = {}

    try:
        if active_splits:
            with ExitStack() as stack:
                for raw_name in C.RAW_FILES:
                    raw_path = C.DATA_RAW / raw_name
                    pending = pd.DataFrame()
                    for chunk in pd.read_csv(
                        raw_path,
                        chunksize=CHUNK_ROWS,
                        low_memory=False,
                        dtype={C.ID: str},
                    ):
                        if pending.empty:
                            combined = chunk
                        else:
                            combined = pd.concat([pending, chunk], ignore_index=True)

                        ids = combined[C.ID].astype(str)
                        last_id = ids.iloc[-1]
                        last_start = len(combined) - 1
                        while last_start > 0 and ids.iloc[last_start - 1] == last_id:
                            last_start -= 1

                        complete = combined.iloc[:last_start]
                        pending = combined.iloc[last_start:].copy()
                        counts = _process_complete_rows(
                            complete,
                            completed_ids,
                            process_split_ids,
                            pipe,
                            features,
                            seen_patients,
                            writers,
                            stack,
                            temp_paths,
                        )
                        for split_name, count in counts.items():
                            written_rows[split_name] += count

                    counts = _process_complete_rows(
                        pending,
                        completed_ids,
                        process_split_ids,
                        pipe,
                        features,
                        seen_patients,
                        writers,
                        stack,
                        temp_paths,
                    )
                    for split_name, count in counts.items():
                        written_rows[split_name] += count
            created_temp_paths.update(
                path for name, path in temp_paths.items() if name in active_splits
            )

        for split_name in active_splits:
            missing = split_ids[split_name] - seen_patients[split_name]
            unexpected = seen_patients[split_name] - split_ids[split_name]
            if missing or unexpected:
                raise ValueError(
                    f"{split_name} patient mismatch: missing={len(missing)}, "
                    f"unexpected={len(unexpected)}"
                )
            temp_path = temp_paths.get(split_name)
            if temp_path is None:
                raise ValueError(f"No {split_name} rows were generated.")
            row_counts[split_name] = _validate_feature_table(
                temp_path, split_ids[split_name], expected_columns
            )
            if row_counts[split_name] != written_rows[split_name]:
                raise ValueError(f"Generated {split_name} row count is inconsistent.")

        if test_path.stat().st_size != test_stat.st_size \
                or test_path.stat().st_mtime_ns != test_stat.st_mtime_ns:
            raise RuntimeError("test.parquet changed during recovery; refusing to finalize.")

        for split_name, temp_path in temp_paths.items():
            target_path = targets[split_name]
            if target_path.exists():
                raise FileExistsError(
                    f"{target_path} appeared during recovery; refusing to overwrite it."
                )
            os.rename(temp_path, target_path)
            row_counts[split_name] = _validate_feature_table(
                target_path, split_ids[split_name], expected_columns
            )

        for split_name in ("train", "val"):
            print(
                f"{split_name}: {len(split_ids[split_name])} patients, "
                f"{row_counts[split_name]} rows, {len(features)} features"
            )
        print("Recovery complete. Existing test.parquet and split/model/result files were not written.")
    finally:
        for temp_path in created_temp_paths:
            if temp_path.exists():
                temp_path.unlink()


def main() -> None:
    _write_split_files()


if __name__ == "__main__":
    main()

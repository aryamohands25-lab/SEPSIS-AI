"""Unit test: features at hour t must not change if all hours after t are deleted."""
import numpy as np
import pandas as pd
import pytest
from src import config as C
from src.features import build_features, clean

N_PATIENTS, CUTS = 12, 3


@pytest.fixture(scope="module")
def sample():
    path = C.DATA_RAW / C.RAW_FILES[0]
    if not path.exists():
        pytest.skip("raw data not available")
    df = pd.read_csv(path, nrows=60000)
    ids = df[C.ID].unique()[:N_PATIENTS]
    df, _ = clean(df[df[C.ID].isin(ids)])
    return df


def test_features_use_only_past(sample):
    _, full = build_features(sample)
    full.index = pd.MultiIndex.from_frame(sample[[C.ID, C.TIME]].reset_index(drop=True))
    rng = np.random.RandomState(0)
    for pid, g in sample.groupby(C.ID):
        for _ in range(CUTS):
            cut = int(rng.randint(2, len(g)))                   # keep rows [0, cut)
            trunc = g.iloc[:cut]
            _, Xt = build_features(trunc)
            last_hour = trunc[C.TIME].iloc[-1]
            a = Xt.iloc[-1].to_numpy(np.float64)
            b = full.loc[(pid, last_hour)].to_numpy(np.float64)
            assert np.allclose(a, b, equal_nan=True), f"future leakage for {pid} at hour {last_hour}"


def test_split_is_by_patient():
    tr = set((C.DATA_SPLITS / "train_ids.txt").read_text().split())
    va = set((C.DATA_SPLITS / "val_ids.txt").read_text().split())
    te = set((C.DATA_SPLITS / "test_ids.txt").read_text().split())
    assert not (tr & va) and not (tr & te) and not (va & te)

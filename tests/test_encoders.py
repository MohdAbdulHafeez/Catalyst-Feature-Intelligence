import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from catalyst.encoders import (
    EncoderKind,
    FrequencyCategoricalEncoder,
    HashingCategoricalEncoder,
    OneHotCategoricalEncoder,
    OrdinalCategoricalEncoder,
    TargetMeanCategoricalEncoder,
    create_encoder,
    supported_encoders,
)


@pytest.fixture()
def categorical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["Hyderabad", "Pune", "Delhi", "Hyderabad", "Pune"],
            "segment": ["A", "B", "A", "C", "B"],
        }
    )


def test_registry_is_complete_and_deterministic() -> None:
    assert supported_encoders() == (
        EncoderKind.ONE_HOT,
        EncoderKind.ORDINAL,
        EncoderKind.FREQUENCY,
        EncoderKind.TARGET_MEAN,
        EncoderKind.HASHING,
    )
    assert isinstance(create_encoder("frequency"), FrequencyCategoricalEncoder)


def test_one_hot_handles_unseen_categories(categorical_frame: pd.DataFrame) -> None:
    encoder = OneHotCategoricalEncoder(sparse_output=False)
    encoder.fit(categorical_frame)
    transformed = encoder.transform(pd.DataFrame({"city": ["Mumbai"], "segment": ["new"]}))
    assert transformed.shape == (1, 6)
    assert np.isfinite(transformed).all()


def test_ordinal_encodes_unknown_and_missing(categorical_frame: pd.DataFrame) -> None:
    encoder = OrdinalCategoricalEncoder()
    encoder.fit(categorical_frame)
    transformed = encoder.transform(
        pd.DataFrame({"city": ["Mumbai", None], "segment": ["A", "new"]})
    )
    assert transformed.shape == (2, 2)
    assert np.isfinite(transformed).all()
    assert -1.0 in transformed


def test_frequency_encoder_uses_training_distribution_only(
    categorical_frame: pd.DataFrame,
) -> None:
    encoder = FrequencyCategoricalEncoder()
    encoder.fit(categorical_frame)
    transformed = encoder.transform(
        pd.DataFrame({"city": ["Hyderabad", "Mumbai"], "segment": ["C", "new"]})
    )
    assert transformed[0, 0] == pytest.approx(2 / 5)
    assert transformed[1, 0] == 0.0


def test_target_encoder_is_training_only() -> None:
    train = pd.DataFrame({"city": ["A", "A", "B", "B"]})
    y_train = pd.Series([0.0, 0.0, 1.0, 1.0])
    encoder = TargetMeanCategoricalEncoder(smoothing=1.0, min_samples_leaf=1)
    encoder.fit(train, y_train)
    transformed = encoder.transform(pd.DataFrame({"city": ["A", "B", "C"]}))
    assert transformed[0, 0] < transformed[1, 0]
    assert transformed[2, 0] == pytest.approx(0.5)


def test_target_encoder_requires_target() -> None:
    encoder = TargetMeanCategoricalEncoder()
    with pytest.raises(ValueError, match="requires y"):
        encoder.fit(pd.DataFrame({"city": ["A", "B"]}))


def test_target_encoder_pipeline_is_cv_safe() -> None:
    frame = pd.DataFrame({"city": np.tile(["A", "B", "C", "D"], 30)})
    y = np.array([0, 0, 1, 1] * 30)
    pipeline = Pipeline(
        [
            ("target_encoder", TargetMeanCategoricalEncoder(smoothing=5.0)),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    scores = cross_val_score(pipeline, frame, y, cv=5, scoring="accuracy")
    assert scores.shape == (5,)
    assert np.isfinite(scores).all()


def test_hashing_has_fixed_width_and_handles_unseen_values(
    categorical_frame: pd.DataFrame,
) -> None:
    encoder = HashingCategoricalEncoder(n_features=16)
    encoder.fit(categorical_frame)
    transformed = encoder.transform(pd.DataFrame({"city": ["brand_new"], "segment": ["unknown"]}))
    assert transformed.shape == (1, 16)
    assert np.isfinite(transformed.toarray()).all()


def test_feature_names_match_contract(categorical_frame: pd.DataFrame) -> None:
    one_hot = OneHotCategoricalEncoder()
    one_hot.fit(categorical_frame)
    assert len(one_hot.get_feature_names_out()) >= 2

    hashing = HashingCategoricalEncoder(n_features=8)
    hashing.fit(categorical_frame)
    assert hashing.get_feature_names_out().tolist() == [f"hash_{i}" for i in range(8)]

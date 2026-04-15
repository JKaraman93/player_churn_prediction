import pytest

from bet.models.logistic_regression import add_class_weight
from bet.models.logistic_regression import compute_metrics

def test_add_class_weight(spark):

    data = [(True, ), (False,), (True,)]
    df = spark.createDataFrame(data, ["next_7d_churn"])
    
    result_df = add_class_weight(df, weight_for_churn=3.0)

    result = [row["class_weight"] for row in result_df.collect()]

    assert result == [3.0, 1.0 , 3.0]
    assert result_df.count() == df.count()


def test_compute_metrics_raises_on_invalid_threshold(spark):
    data = [(0.8, 1 ), (0.3, 0), (0.5 , 0)]
    df = spark.createDataFrame(data, ["p_churn", "next_7d_churn_idx"])

    with pytest.raises(ValueError):
        compute_metrics(df, threshold=1.5)




@pytest.mark.parametrize("threshold, expected_precision", [
    (0.5, 1.0),
    (0.9, 0.0),
    (0.2, 0.67),
])

def test_compute_metrics_multiple_thresholds(spark, threshold, expected_precision):
    # Arrange: 3 rows — 2 churn, 1 non-churn
    data = [(0.8, 1, "2024-01-01"), (0.3, 0, "2024-01-01"), (0.4 , 1, "2024-01-01")]
    df = spark.createDataFrame(data, ["p_churn", "next_7d_churn_idx", "reference_date"])
    precision, recall,_, _, _= compute_metrics(df, threshold=threshold)
    assert precision == pytest.approx(expected_precision)
    #assert recall == pytest.approx(0.5)



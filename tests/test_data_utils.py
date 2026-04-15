from bet.utils.data_utils  import extract_player_idx_from_id

def test_extract_player_idx_from_id(spark): 
    data = [("player_0001",), ("player_0023",), ("player_1234",)]
    df = spark.createDataFrame(data, ["player_id"])
    
    result_df = extract_player_idx_from_id(df)
    result = [row["player_idx"] for row in result_df.collect()]

    assert "player_id" not in result_df.columns
    assert result == [1, 23, 1234]



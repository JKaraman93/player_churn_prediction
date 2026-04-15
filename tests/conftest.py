import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession once for the whole test session."""
    return SparkSession.builder.master("local[1]").appName("test").getOrCreate()
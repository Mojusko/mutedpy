import pytest
import getpass
import torch
import pandas as pd
from mutedpy.protein_learning.embeddings.lookupmap import LookUpMapPostgres

@pytest.fixture
def postgres_lookup():
    # Prompt for password
    password = getpass.getpass("Enter PostgreSQL password: ")

    # Initialize LookUpMapPostgres
    return LookUpMapPostgres(
        server="mmutny@matroid2.inf.ethz.ch",
        credentials=password,
        database="embeddings_db",
        project="esm2-650",
        data=None,
        embedding_name="embedding"
    )

def test_connection(postgres_lookup):
    try:
        postgres_lookup.connect(verbose=True)
        assert postgres_lookup.conn is not None, "Connection failed."
    finally:
        postgres_lookup.close(verbose=True)
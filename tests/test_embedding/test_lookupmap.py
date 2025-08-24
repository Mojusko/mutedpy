import pytest
import torch
import pandas as pd
from mutedpy.protein_learning.embeddings.lookupmap_static import LookUpMap

@pytest.fixture
def sample_data(tmp_path):
    # Create a temporary CSV file with sample data
    data = {
        'seq': ['A', 'B', 'C'],
        'embedding1': [1.0, 2.0, 3.0],
        'embedding2': [4.0, 5.0, 6.0]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def lookup_map(sample_data):
    return LookUpMap(data=sample_data)

def test_initialization(lookup_map):
    assert lookup_map.N == 3
    assert lookup_map.d == 1
    assert len(lookup_map.feature_names) == 2

def test_get_indices(lookup_map):
    x = torch.tensor([[1, 2], [3, 4], [5, 6]])
    indices = lookup_map._get_indices(x)
    assert len(indices) == 3

def test_restrict_by_std(lookup_map):
    lookup_map.restrict_by_std(std=0.5)
    assert len(lookup_map.feature_names) <= 2

def test_restrict_by_name(lookup_map):
    lookup_map.restrict_by_name(names=["embedding1"])
    assert lookup_map.feature_names == ["embedding1"]

def test_pca(lookup_map):
    lookup_map.pca(expl_var=0.9)
    assert lookup_map.m <= 2

def test_standardize(lookup_map):
    lookup_map.standardize()
    mean = torch.mean(lookup_map.phi, dim=0)
    std = torch.std(lookup_map.phi, dim=0)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-6)

def test_normalize(lookup_map):
    lookup_map.normalize()
    assert torch.all(lookup_map.phi >= -1) and torch.all(lookup_map.phi <= 1)

def test_l2_normalize(lookup_map):
    lookup_map.l2_normalize()
    norms = torch.norm(lookup_map.phi, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

def test_embed(lookup_map):
    x = torch.tensor([[1, 2], [3, 4]])
    with pytest.raises(AssertionError):
        lookup_map.embed(x)

def test_embed_seq(lookup_map):
    seqs = ['A', 'B']
    embeddings = lookup_map.embed_seq(seqs)
    assert embeddings.size(0) == len(seqs)
    assert embeddings.size(1) == lookup_map.m

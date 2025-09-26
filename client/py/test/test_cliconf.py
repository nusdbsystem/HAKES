import os
import pytest

from hakesclient import ClientConfig


def test_load_from_file_missing_fields():
    with pytest.raises(ValueError):
        ClientConfig.from_file("testdata/empty.json")


def test_load_from_file():
    conf = ClientConfig.from_file("testdata/config.json")

    assert conf.search_worker_addrs == ["127.0.0.1:2800"]
    assert conf.preferred_search_worker == 0
    assert conf.hakes_addr == "127.0.0.1:8080"
    assert conf.embed_endpoint_type == "openai"
    assert conf.embed_endpoint_config == ""
    assert conf.store_type == "hakes"
    assert conf.store_addr == "127.0.0.1:2600"


def test_to_dict():
    conf = ClientConfig.from_file("testdata/config.json")

    assert conf.to_dict() == {
        "search_worker_addrs": ["127.0.0.1:2800"],
        "preferred_search_worker": 0,
        "hakes_addr": "127.0.0.1:8080",
        "embed_endpoint_type": "openai",
        "embed_endpoint_config": "",
        "store_type": "hakes",
        "store_addr": "127.0.0.1:2600",
    }

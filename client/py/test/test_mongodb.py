import pytest
from hakesclient.extensions.mongodb import MongoDB
import traceback

def mongodb_available():
    try:
        client = MongoDB("mongodb://localhost:27017", db_name="test_hakes", collection_name="test_kv_store")
        return client.connected()
    except Exception:
        # print the exception
        print(traceback.format_exc())
        return False

pytestmark = pytest.mark.skipif(not mongodb_available(), reason="MongoDB server is not available on localhost:27017")

def test_mongodb_store_integration():
    db_name = "test_hakes"
    collection_name = "test_kv_store"
    store = MongoDB("mongodb://localhost:27017", db_name=db_name, collection_name=collection_name)
    store.collection.delete_many({})  # Clean up before test
    store.counters.delete_many({"_id": collection_name})

    # Batched put with auto-increment xids
    keys = [f"k{i}" for i in range(10)]
    values = [f"v{i}".encode() for i in range(10)]
    ok, xids = store.put(keys, values, None)
    assert ok
    assert len(xids) == len(keys)
    # All xids should be 8 bytes and unique
    assert all(isinstance(xid, bytes) and len(xid) == 8 for xid in xids)
    assert len(set(xids)) == len(xids)

    # Batched get_by_keys
    out_values, out_xids = store.get_by_keys(keys)
    assert out_values == values
    assert out_xids == xids

    # Batched get_by_ids
    out_by_ids = store.get_by_ids(xids)
    assert out_by_ids == values

    # Batched put with explicit xids
    custom_xids = [b"customid%02d" % i for i in range(10, 15)]
    custom_keys = [f"k{i}" for i in range(10, 15)]
    custom_values = [f"v{i}".encode() for i in range(10, 15)]
    ok, used_xids = store.put(custom_keys, custom_values, custom_xids)
    assert ok
    assert used_xids == custom_xids
    # get_by_keys for mixed keys
    all_keys = keys + custom_keys
    all_values, all_xids = store.get_by_keys(all_keys)
    assert all_values == values + custom_values
    assert all_xids == xids + custom_xids

    # get_by_ids for mixed xids
    out = store.get_by_ids(xids + custom_xids)
    assert out == values + custom_values

    # Edge case: missing keys
    missing_keys = ["notfound1", "notfound2"]
    vals, xids_missing = store.get_by_keys(missing_keys)
    assert vals == [b"", b""]
    assert xids_missing == [b"", b""]

    # Edge case: missing xids
    fake_xids = [b"fakeid01", b"fakeid02"]
    out = store.get_by_ids(fake_xids)
    assert out == [b"", b""]

    # Clean up
    store.collection.delete_many({})
    store.counters.delete_many({"_id": collection_name})

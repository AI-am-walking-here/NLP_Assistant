from grounded.config import load_config
from grounded.data.s3_pull import build_botocore_config, build_transfer_config


def test_load_config_includes_s3_transfer_settings():
    config = load_config()
    transfer = config.sources.arxiv_s3.transfer
    assert transfer.multipart_threshold_mb == 8
    assert transfer.multipart_chunksize_mb == 16
    assert transfer.max_concurrency == 10
    assert transfer.max_pool_connections == 32
    assert config.sources.arxiv_s3.max_workers == 4


def test_build_transfer_config_maps_megabytes_to_bytes():
    config = load_config()
    transfer_cfg = build_transfer_config(config.sources.arxiv_s3.transfer)
    assert transfer_cfg.multipart_threshold == 8 * 1024 * 1024
    assert transfer_cfg.multipart_chunksize == 16 * 1024 * 1024
    assert transfer_cfg.max_concurrency == 10


def test_build_botocore_config_sets_pool_size():
    config = load_config()
    boto_cfg = build_botocore_config(config.sources.arxiv_s3.transfer)
    assert boto_cfg.max_pool_connections == 32

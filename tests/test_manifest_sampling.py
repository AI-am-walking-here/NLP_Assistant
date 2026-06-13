from grounded.data.filter import ManifestEntry, build_filtered_manifest


def test_build_filtered_manifest_samples_within_budget():
    entries = [
        ManifestEntry("src/arXiv_src_2401_001.tar", "2401.00001", "2401.00010", "2401", 100),
        ManifestEntry("src/arXiv_src_2401_002.tar", "2401.00011", "2401.00020", "2401", 100),
        ManifestEntry("src/arXiv_src_2401_003.tar", "2401.00021", "2401.00030", "2401", 100),
    ]
    target_ids = {"2401.00001", "2401.00011", "2401.00021"}

    payload = build_filtered_manifest(
        entries,
        target_ids,
        hard_cap_usd=1.0,
        egress_per_gb_usd=0.0,
        get_request_usd=0.6,
        random_seed=7,
    )

    assert payload["stats"]["selection_mode"] == "random_budget_sample"
    assert payload["stats"]["num_tarballs"] == 1
    assert payload["stats"]["estimated_cost_usd"] == 0.6
    assert len(payload["arxiv_id_to_tarball"]) == 1

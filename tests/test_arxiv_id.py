from grounded.data.filter import arxiv_id_from_member, in_id_range, parse_arxiv_id


def test_parse_arxiv_id():
    parsed = parse_arxiv_id("2401.12345")
    assert parsed is not None
    assert parsed.yymm == "2401"
    assert parsed.number == 12345


def test_in_id_range():
    assert in_id_range("2401.05000", "2401.00001", "2401.99999")
    assert not in_id_range("2402.05000", "2401.00001", "2401.99999")


def test_arxiv_id_from_member():
    assert arxiv_id_from_member("2401.12345/main.tex") == "2401.12345"
    assert arxiv_id_from_member("2401.12345.tar.gz") == "2401.12345"
    assert arxiv_id_from_member("2401.12345.gz") == "2401.12345"
    assert arxiv_id_from_member("bulk/2401.12345.gz") == "2401.12345"

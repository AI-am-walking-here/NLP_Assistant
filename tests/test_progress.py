from grounded.progress import ByteProgressReporter, CountProgressReporter, format_bytes, format_duration


def test_format_bytes():
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(3 * 1024**2) == "3.0 MB"


def test_byte_progress_reporter_finish(capsys):
    reporter = ByteProgressReporter("test", total_bytes=1000, min_interval_s=0.0)
    reporter.update(250)
    reporter.update(750)
    reporter.finish()
    output = capsys.readouterr().out
    assert "[test]" in output
    assert "Complete:" in output


def test_count_progress_reporter_finish(capsys):
    reporter = CountProgressReporter("scan", total=2, unit="files", min_interval_s=0.0)
    reporter.update(1, detail="a.jsonl")
    reporter.update(1, detail="b.jsonl")
    reporter.finish(detail="done")
    output = capsys.readouterr().out
    assert "1/2 files" in output
    assert "Complete: 2 files" in output


def test_format_duration():
    assert format_duration(45) == "45s"
    assert format_duration(75) == "1m 15s"

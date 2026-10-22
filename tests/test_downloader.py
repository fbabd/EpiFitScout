"""RCSB download behaviour: certificates, timeouts, retries, and the PDB→mmCIF
fallback that depends on how a 404 is reported.

The certificate case is the one that bit in practice: a Python built outside the
distribution (uv, conda) looks for CA certificates in a Debian-style path, so on
a RHEL-family cluster every HTTPS request failed with CERTIFICATE_VERIFY_FAILED
even though ``curl`` worked.
"""

from __future__ import annotations

import io
import ssl
import urllib.error
from pathlib import Path

import pytest

from epifitscout.db import rcsb_downloader as rd
from epifitscout.db.rcsb_downloader import RcsbDownloader, RcsbDownloadError


class _Response(io.BytesIO):
    """Minimal stand-in for the object urlopen returns."""

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rd.time, "sleep", lambda s: None)


class TestSslContext:
    def _record(self, monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
        seen: list[str | None] = []

        def fake(cafile: str | None = None, **kwargs: object) -> str:
            seen.append(cafile)
            return "ctx"

        monkeypatch.setattr(rd.ssl, "create_default_context", fake)
        return seen

    def test_ssl_cert_file_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mine = tmp_path / "mine.pem"
        mine.write_text("x")
        monkeypatch.setenv("SSL_CERT_FILE", str(mine))
        seen = self._record(monkeypatch)
        rd._ssl_context()
        assert seen == [str(mine)]

    def test_falls_back_to_the_platform_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = tmp_path / "ca-bundle.crt"
        bundle.write_text("x")
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.setattr(rd, "_CA_BUNDLES", ("/no/such/file", str(bundle)))
        seen = self._record(monkeypatch)
        rd._ssl_context()
        assert seen == [str(bundle)]

    def test_real_context_is_buildable_here(self) -> None:
        assert isinstance(rd._ssl_context(), ssl.SSLContext)


class TestFetch:
    def _downloader(self, tmp_path: Path) -> RcsbDownloader:
        return RcsbDownloader(cache_dir=tmp_path / "cache")

    def test_success_writes_the_file_and_passes_a_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(url: str, timeout: float, context: object) -> _Response:
            captured.update(url=url, timeout=timeout)
            return _Response(b"ATOM      1  N   ALA A   1\n")

        monkeypatch.setattr(rd.urllib.request, "urlopen", fake_urlopen)
        sf = self._downloader(tmp_path).download("1abc")
        assert sf.path.read_bytes().startswith(b"ATOM")
        assert sf.fmt == "pdb"
        assert captured["timeout"] and float(captured["timeout"]) > 0  # type: ignore[arg-type]
        assert not list((tmp_path / "cache").glob("*.tmp"))

    def test_transient_failures_are_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def flaky(url: str, timeout: float, context: object) -> _Response:
            calls["n"] += 1
            if calls["n"] < 3:
                raise urllib.error.URLError("connection reset")
            return _Response(b"ATOM\n")

        monkeypatch.setattr(rd.urllib.request, "urlopen", flaky)
        self._downloader(tmp_path).download("1abc")
        assert calls["n"] == 3

    def test_gives_up_after_the_attempt_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def down(url: str, timeout: float, context: object) -> _Response:
            calls["n"] += 1
            raise urllib.error.URLError("unreachable")

        monkeypatch.setattr(rd.urllib.request, "urlopen", down)
        with pytest.raises(RcsbDownloadError, match="after 3 attempts"):
            self._downloader(tmp_path).download("1abc")
        assert calls["n"] == 3

    def test_certificate_failure_explains_the_fix_and_is_not_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def bad_cert(url: str, timeout: float, context: object) -> _Response:
            calls["n"] += 1
            raise urllib.error.URLError(ssl.SSLCertVerificationError("unable to get issuer"))

        monkeypatch.setattr(rd.urllib.request, "urlopen", bad_cert)
        with pytest.raises(RcsbDownloadError, match="SSL_CERT_FILE"):
            self._downloader(tmp_path).download("1abc")
        assert calls["n"] == 1, "a certificate problem will not fix itself on retry"

    def test_404_falls_back_to_mmcif_without_retrying(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Large entries have no PDB file; the fallback keys off the 'HTTP 404' text."""
        urls: list[str] = []

        def serve(url: str, timeout: float, context: object) -> _Response:
            urls.append(url)
            if url.endswith(".pdb"):
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]
            return _Response(b"data_1ABC\n")

        monkeypatch.setattr(rd.urllib.request, "urlopen", serve)
        sf = self._downloader(tmp_path).download("1abc")
        assert sf.fmt == "cif"
        assert [u.rsplit(".", 1)[1] for u in urls] == ["pdb", "cif"]

    def test_other_http_errors_are_not_retried_or_masked_as_404(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def server_error(url: str, timeout: float, context: object) -> _Response:
            calls["n"] += 1
            raise urllib.error.HTTPError(url, 503, "Unavailable", {}, None)  # type: ignore[arg-type]

        monkeypatch.setattr(rd.urllib.request, "urlopen", server_error)
        with pytest.raises(RcsbDownloadError, match="HTTP 503"):
            self._downloader(tmp_path).download("1abc")
        assert calls["n"] == 1

"""Public data acquisition never inherits credentials or connects to private hosts."""

import hashlib
import os
import socket
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from symplex.agents.tools import REGISTRY, ToolContext
from symplex.connectors.compute import read_blob, run
from symplex.connectors.public_artifacts import (
    PublicArtifactPolicy, PublicArtifactRequest, fetch,
)
from symplex.connectors.registry import ConnectorInput, execute
from symplex.core.contracts import Invalid
from symplex.infrastructure.storage import Budget, BudgetExhausted, Store


class PublicArtifactTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(directory.name)
        self.budget = Budget(self.store)
        self.problem = self.store.put("workspace_problem", {"question": "Inspect actual source observations"})
        self.allowed = PublicArtifactPolicy(allowed_hosts=["data.example.org", "archive.example.org"])
        self.request = PublicArtifactRequest(url="https://data.example.org/observations.csv", filename="observations.csv")
        self.raw = b"time,value\n0,1.25\n1,2.50\n"
        self.calls = []
        dns = patch("symplex.connectors.public_artifacts.socket.getaddrinfo", return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443))
        ])
        self.dns = dns.start()
        self.addCleanup(dns.stop)
        transport = patch("symplex.connectors.public_artifacts.httpx.HTTPTransport", side_effect=lambda **kwargs: httpx.MockTransport(self.respond))
        self.transport = transport.start()
        self.addCleanup(transport.stop)
        self.response = lambda request: httpx.Response(200, headers={"content-type": "text/csv"}, stream=httpx.ByteStream(self.raw))

    def respond(self, request):
        self.calls.append(request)
        return self.response(request)

    def fetch(self, request=None, allowed=None):
        return fetch(self.store, self.budget, self.problem, request or self.request, allowed=allowed or self.allowed)

    def assert_not_admitted(self):
        self.assertFalse(self.store.list("file_blob"))
        self.assertFalse(self.store.list("binary_context"))
        self.assertEqual(self.store.list("artifact_fetch_failure")[-1]["data"]["status"], "not_admitted")

    def test_fetch_persists_exact_bytes_digest_provenance_and_structural_profile(self):
        self.request.expected_sha256 = hashlib.sha256(self.raw).hexdigest()
        self.request.expected_csv_columns = ["time", "value"]
        result = self.fetch()
        self.assertEqual(result["sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(result["inspection"]["rows"], 2)
        self.assertEqual(result["inspection"]["schema_status"], "requested_columns_matched")
        self.assertEqual(result["basis"], "external_source")
        self.assertEqual(result["status"], "retrieved_unvalidated")
        self.assertIn("unverified", result["scope"])
        blob = self.store.get(result["blob_id"])
        self.assertEqual(blob["parent"], self.problem)
        self.assertEqual(blob["data"]["basis"], "external_source")
        self.assertEqual(read_blob(self.store, blob), self.raw)
        binary = self.store.get(result["binary_context_id"])
        self.assertEqual(binary["data"]["blob_id"], blob["id"])
        self.assertEqual(binary["data"]["source_id"], result["source_id"])
        self.assertEqual(result["source_url"], self.request.url)
        self.assertTrue(result["retrieved_at"])
        self.assertEqual(self.budget.snapshot()["evidence_requests"]["used_or_reserved"], 1)

    def test_transport_pins_public_ip_keeps_tls_hostname_and_ignores_environment_credentials(self):
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://private-proxy:8080", "NETRC": "/private/secret.netrc"}):
            self.fetch()
        sent = self.calls[0]
        self.assertEqual(sent.url.host, "93.184.216.34")
        self.assertEqual(sent.headers["host"], "data.example.org")
        self.assertEqual(sent.extensions["sni_hostname"], "data.example.org")
        self.assertEqual(sent.headers["accept-encoding"], "identity")
        self.assertNotIn("authorization", sent.headers)
        self.assertNotIn("cookie", sent.headers)
        self.transport.assert_called_once_with(verify=True, trust_env=False, retries=0)
        self.assertEqual(self.dns.call_count, 1)

    def test_unsafe_source_urls_fail_before_dns_http_or_budget_reservation(self):
        unsafe = ["http://data.example.org/a.csv", "https://localhost/a.csv",
                  "https://127.0.0.1/a.csv", "https://[::1]/a.csv",
                  "https://user:secret@data.example.org/a.csv", "https://data.example.org:8443/a.csv",
                  "https://data.example.org.evil.org/a.csv", "https://data.example.org./a.csv",
                  "https://data.example.org/a.csv?api_key=secret", "https://data.example.org/a.csv#fragment",
                  "https://data.example.org\\@evil.org/a.csv", "https://data.example.org/a\n.csv"]
        for url in unsafe:
            with self.subTest(url=url), self.assertRaises(Invalid):
                self.fetch(self.request.model_copy(update={"url": url}))
        self.assertFalse(self.calls)
        self.assertEqual(self.dns.call_count, 0)
        self.assertEqual(self.budget.snapshot()["evidence_requests"]["used_or_reserved"], 0)

    def test_private_or_mixed_dns_answers_fail_closed_before_http(self):
        for addresses in (["127.0.0.1"], ["10.0.0.1"], ["169.254.169.254"], ["::1"],
                          ["fc00::1"], ["224.0.0.1"], ["93.184.216.34", "192.168.1.1"]):
            self.dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 443)) for ip in addresses]
            with self.subTest(addresses=addresses), self.assertRaises(Invalid):
                self.fetch()
        self.assertFalse(self.calls)
        self.assertEqual(self.budget.snapshot()["evidence_requests"]["used_or_reserved"], 0)

    def test_redirect_is_validated_re_pinned_and_does_not_forward_cookies(self):
        def respond(request):
            if len(self.calls) == 1:
                return httpx.Response(302, headers={"location": "https://archive.example.org/copy.csv", "set-cookie": "private=secret"})
            return httpx.Response(200, stream=httpx.ByteStream(self.raw))
        self.response = respond
        result = self.fetch()
        self.assertEqual(result["retrieved_url"], "https://archive.example.org/copy.csv")
        self.assertEqual(len(result["redirects"]), 1)
        self.assertEqual(self.calls[1].headers["host"], "archive.example.org")
        self.assertEqual(self.calls[1].extensions["sni_hostname"], "archive.example.org")
        self.assertNotIn("cookie", self.calls[1].headers)
        self.assertEqual(self.budget.snapshot()["evidence_requests"]["used_or_reserved"], 2)

    def test_private_redirect_is_rejected_without_following_it(self):
        self.response = lambda request: httpx.Response(302, headers={"location": "https://127.0.0.1/internal.csv"})
        with self.assertRaises(Invalid):
            self.fetch()
        self.assertEqual(len(self.calls), 1)
        self.assert_not_admitted()
        self.assertEqual(self.store.list("artifact_fetch_failure")[-1]["data"]["phase"], "redirect_preflight")

    def test_redirect_loop_is_bounded(self):
        self.response = lambda request: httpx.Response(302, headers={"location": "/again.csv"})
        with self.assertRaises(Invalid):
            self.fetch()
        self.assertEqual(len(self.calls), 4)
        self.assert_not_admitted()

    def test_stream_and_content_length_limits_prevent_admission(self):
        self.response = lambda request: httpx.Response(200, headers={"content-length": "2000001"}, stream=httpx.ByteStream(self.raw))
        with self.assertRaises(Invalid):
            self.fetch()
        self.assert_not_admitted()
        self.response = lambda request: httpx.Response(200, stream=httpx.ByteStream(b"a,b\n" + b"1,2\n" * 100))
        with self.assertRaises(Invalid):
            self.fetch(allowed=PublicArtifactPolicy(allowed_hosts=self.allowed.allowed_hosts, max_bytes=32))
        self.assert_not_admitted()

    def test_compressed_or_html_error_pages_cannot_be_admitted_as_csv(self):
        for headers, raw in [({"content-encoding": "gzip"}, self.raw), ({"content-type": "text/html"}, self.raw),
                             ({}, b"<!doctype html><html>Login required</html>")]:
            self.response = lambda request, headers=headers, raw=raw: httpx.Response(200, headers=headers, stream=httpx.ByteStream(raw))
            with self.subTest(headers=headers), self.assertRaises(Invalid):
                self.fetch()
            self.assert_not_admitted()

    def test_digest_and_schema_mismatches_are_not_admitted(self):
        for update in ({"expected_sha256": "0" * 64}, {"expected_csv_columns": ["invented", "value"]}):
            with self.subTest(update=update), self.assertRaises(Invalid):
                self.fetch(self.request.model_copy(update=update))
            self.assert_not_admitted()
        self.raw = b"time,value\n0,1,unexpected\n"
        with self.assertRaises(Invalid):
            self.fetch()
        self.assert_not_admitted()

    def test_strict_request_schema_rejects_headers_paths_and_large_policy(self):
        for update in ({"headers": {"Authorization": "secret"}}, {"filename": "../data.csv"},
                       {"filename": "script.py"}, {"expected_sha256": "not-a-hash"}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                PublicArtifactRequest.model_validate(dict(self.request.model_dump(), **update))
        with self.assertRaises(ValidationError):
            PublicArtifactPolicy(allowed_hosts=["*.example.org"])
        with self.assertRaises(ValidationError):
            PublicArtifactPolicy(allowed_hosts=["data.example.org"], max_bytes=5000001)
        self.assertFalse(self.calls)

    def test_http_failure_has_a_durable_failure_record_without_response_or_secret_payload(self):
        self.response = lambda request: httpx.Response(403, text="private server content")
        with self.assertRaises(Invalid) as raised:
            self.fetch()
        self.assert_not_admitted()
        failure = self.store.list("artifact_fetch_failure")[-1]
        self.assertIn(failure["id"], str(raised.exception))
        self.assertNotIn("private server content", str(failure))

    def test_budget_exhaustion_is_preserved_and_no_http_request_occurs(self):
        self.budget.reserve(evidence_requests=20)
        with self.assertRaises(BudgetExhausted):
            self.fetch()
        self.assertFalse(self.calls)

    def test_registry_permission_and_typed_connector_dispatch(self):
        context = ToolContext(self.store, self.budget, None, self.problem, None, [], [])
        action = SimpleNamespace(instruction=self.request.model_dump_json())
        with self.assertRaises(Invalid):
            REGISTRY.execute("fetch_artifact", context, action, {"artifacts.read"})
        self.assertFalse(self.calls)
        with patch("symplex.connectors.public_artifacts.policy", return_value=self.allowed):
            inventory = REGISTRY.execute("discover_datasets", context, action, {"artifacts.read"})
            result = REGISTRY.execute("fetch_artifact", context, action, {"network.dataset"})
            response = execute(self.store, self.budget, None, ConnectorInput(
                connector="public_artifact", problem_id=self.problem, **self.request.model_dump()), {"network.dataset"})
        self.assertIn(result["blob_id"], context.result_ids)
        self.assertEqual(response["result"]["basis"], "external_source")
        self.assertEqual(inventory["public_artifact_access"]["allowed_hosts"], self.allowed.allowed_hosts)

    def test_stale_problem_fails_before_source_preflight_or_network(self):
        self.store.invalidate(self.problem)
        with self.assertRaises(Invalid):
            self.fetch()
        self.assertFalse(self.calls)
        self.assertEqual(self.dns.call_count, 0)

    def test_downloaded_raw_bytes_travel_to_existing_hosted_compute_transport(self):
        result = self.fetch()
        uploads = []

        def upload(file, purpose):
            uploads.append(file)
            return SimpleNamespace(id="remote-" + str(len(uploads)))

        call = SimpleNamespace(type="code_interpreter_call", model_dump=lambda: {"status": "completed", "container_id": "fake"})
        provider = SimpleNamespace(
            client=SimpleNamespace(files=SimpleNamespace(create=upload, delete=lambda ident: None)),
            _call=lambda *args, **kwargs: SimpleNamespace(id="fake-response", model="fake", output=[call], output_text="Source input inspected"),
        )
        with patch("symplex.connectors.compute.collect_outputs", return_value="fake_package"):
            run(self.store, self.budget, provider, self.problem, "Inspect source bytes")
        self.assertIn(self.raw, [raw for _, raw in uploads])
        sources = self.store.list("remote_file")
        actual = next(r for r in sources if r["data"].get("blob_id") == result["blob_id"])
        self.assertEqual(actual["data"]["content_sha256"], result["sha256"])


if __name__ == "__main__":
    unittest.main()

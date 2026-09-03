from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import provision_v2


class ProvisionV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = self.root / ".env"
        self.config = self.root / "v2-clients.yml"
        self.router = self.root / "corpus-router.yml"
        self.env.write_text("EXISTING=value\n", encoding="utf-8")
        os.chmod(self.env, 0o600)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_install_is_private_idempotent_and_preserves_existing_env(self) -> None:
        with patch(
            "provision_v2.secrets.token_hex",
            side_effect=["a" * 64, "b" * 64, "c" * 64],
        ):
            first = provision_v2.install(self.env, self.config, self.router)
        second = provision_v2.install(self.env, self.config, self.router)
        self.assertEqual(first["tokens_created"], 3)
        self.assertEqual(second["tokens_created"], 0)
        content = self.env.read_text(encoding="utf-8")
        self.assertIn("EXISTING=value", content)
        self.assertEqual(content.count("KB_V2_TOKEN_MCP_LOCAL="), 1)
        self.assertEqual(content.count("KB_V2_TOKEN_KB_CLI_LOCAL="), 1)
        self.assertEqual(content.count("KB_V2_TOKEN_KERRIGAN="), 1)
        self.assertEqual(self.env.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.config.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.router.stat().st_mode & 0o777, 0o600)
        self.assertEqual(yaml.safe_load(self.config.read_text()), {"clients": provision_v2.CLIENTS})
        self.assertEqual(
            provision_v2.check(self.env, self.config, self.router),
            {
                "tokens": 3,
                "clients": 3,
                "router_version": "corpus-router-v1-precalibration",
            },
        )

    def test_install_preserves_all_existing_token_values_including_kerrigan(self) -> None:
        original = (
            "EXISTING=value\n"
            "KB_V2_TOKEN_MCP_LOCAL=" + "a" * 64 + "\n"
            "KB_V2_TOKEN_KB_CLI_LOCAL=" + "b" * 64 + "\n"
            "KB_V2_TOKEN_KERRIGAN=" + "c" * 64 + "\n"
        )
        self.env.write_text(original, encoding="utf-8")
        self.config.write_text(
            yaml.safe_dump({"clients": provision_v2.CLIENTS}, sort_keys=False),
            encoding="utf-8",
        )
        self.router.write_text(
            yaml.safe_dump(provision_v2.ROUTER_CONFIG, sort_keys=False),
            encoding="utf-8",
        )

        with patch("provision_v2.secrets.token_hex") as token_hex:
            result = provision_v2.install(self.env, self.config, self.router)

        token_hex.assert_not_called()
        self.assertEqual(result["tokens_created"], 0)
        self.assertEqual(self.env.read_text(encoding="utf-8"), original)
        self.assertEqual(provision_v2.check(self.env, self.config, self.router)["clients"], 3)

    def test_install_rejects_duplicate_or_unexpected_existing_contract(self) -> None:
        self.env.write_text(
            "KB_V2_TOKEN_MCP_LOCAL=" + "a" * 64 + "\n"
            "KB_V2_TOKEN_MCP_LOCAL=" + "b" * 64 + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            provision_v2.install(self.env, self.config, self.router)

        self.env.write_text("EXISTING=value\n", encoding="utf-8")
        self.config.write_text("clients: {}\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            provision_v2.install(self.env, self.config, self.router)
        self.assertEqual(self.env.read_text(encoding="utf-8"), "EXISTING=value\n")

    def test_install_preflights_router_conflict_before_env_mutation(self) -> None:
        self.router.write_text("router_version: unexpected\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "unexpected router config"):
            provision_v2.install(self.env, self.config, self.router)
        self.assertEqual(self.env.read_text(encoding="utf-8"), "EXISTING=value\n")

    def test_owner_and_runtime_token_contract_match_api_loader(self) -> None:
        with patch("provision_v2.os.geteuid", return_value=os.geteuid() + 1):
            with self.assertRaisesRegex(RuntimeError, "not owned"):
                provision_v2.install(self.env, self.config, self.router)
        self.env.write_text(
            "KB_V2_TOKEN_MCP_LOCAL=" + "!" * 64 + "\n"
            "KB_V2_TOKEN_KB_CLI_LOCAL=" + "b" * 64 + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "valid strong local token"):
            provision_v2.install(self.env, self.config, self.router)


if __name__ == "__main__":
    unittest.main()

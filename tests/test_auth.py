"""Tests pour le gate d'authentification."""
from __future__ import annotations

from unittest.mock import patch

from ui.auth import _check, _password_required


class TestPasswordRequired:
    @patch("ui.auth.settings")
    def test_empty_password_means_no_gate(self, mock_settings) -> None:
        mock_settings.app_password = ""
        assert not _password_required()

    @patch("ui.auth.settings")
    def test_whitespace_password_means_no_gate(self, mock_settings) -> None:
        mock_settings.app_password = "   "
        assert not _password_required()

    @patch("ui.auth.settings")
    def test_non_empty_password_activates_gate(self, mock_settings) -> None:
        mock_settings.app_password = "secret123"
        assert _password_required()


class TestPasswordCheck:
    @patch("ui.auth.settings")
    def test_correct_password_accepted(self, mock_settings) -> None:
        mock_settings.app_password = "secret123"
        assert _check("secret123")

    @patch("ui.auth.settings")
    def test_wrong_password_rejected(self, mock_settings) -> None:
        mock_settings.app_password = "secret123"
        assert not _check("wrong")

    @patch("ui.auth.settings")
    def test_case_sensitive(self, mock_settings) -> None:
        mock_settings.app_password = "Secret"
        assert not _check("secret")
        assert _check("Secret")

    @patch("ui.auth.settings")
    def test_empty_input_rejected(self, mock_settings) -> None:
        mock_settings.app_password = "secret123"
        assert not _check("")

"""Unit tests for i18n message loading and retrieval."""

from pathlib import Path

import pytest
import yaml

from src.i18n import clear_cache, get_message, load_messages
from src.i18n.keys import (
    CHAT_CHATTING_WITH,
    CHAT_EXIT_INSTRUCTIONS,
    CHAT_INPUT_PROMPT,
    CHAT_WELCOME,
    ERROR_CHAIN_NOT_INITIALIZED,
    ERROR_GENERATING_RESPONSE,
    ERROR_GENERIC,
    SOURCES_ITEM_PREFIX_CLI,
    SOURCES_ITEM_SEPARATOR_WEB,
    SOURCES_LABEL_CLI,
    SOURCES_LABEL_WEB,
    SOURCES_NONE,
    SOURCES_SUFFIX_WEB,
    STATUS_REFLECTING_SHORT,
    STATUS_REFLECTING_VERBOSE,
)


class TestLoadMessages:
    """Tests for load_messages function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    def test_loads_english_messages(self) -> None:
        """Should load English messages from en.yaml."""
        messages = load_messages("en")

        assert isinstance(messages, dict)
        assert "chat" in messages
        assert "status" in messages
        assert "sources" in messages
        assert "errors" in messages

    def test_loads_french_messages(self) -> None:
        """Should load French messages from fr.yaml."""
        messages = load_messages("fr")

        assert isinstance(messages, dict)
        assert "chat" in messages
        assert "status" in messages
        assert "sources" in messages
        assert "errors" in messages

    def test_caches_loaded_messages(self) -> None:
        """Should cache messages to avoid re-reading files."""
        messages1 = load_messages("en")
        messages2 = load_messages("en")

        # Should return the same cached object
        assert messages1 is messages2

    def test_clear_cache_reloads_messages(self) -> None:
        """Should reload messages after cache is cleared."""
        messages1 = load_messages("en")
        clear_cache()
        messages2 = load_messages("en")

        # Should be equal but not the same object
        assert messages1 == messages2
        assert messages1 is not messages2

    def test_raises_for_nonexistent_language(self) -> None:
        """Should raise FileNotFoundError for unsupported language."""
        with pytest.raises(FileNotFoundError, match="Locale file not found"):
            load_messages("de")


class TestGetMessage:
    """Tests for get_message function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    def test_gets_simple_message_english(self) -> None:
        """Should retrieve simple English message."""
        result = get_message(CHAT_WELCOME, "en")
        assert result == "Welcome to Luminary!"

    def test_gets_simple_message_french(self) -> None:
        """Should retrieve simple French message."""
        result = get_message(CHAT_WELCOME, "fr")
        assert result == "Bienvenue à Luminary !"

    def test_gets_nested_message(self) -> None:
        """Should retrieve message using dot notation."""
        result = get_message(STATUS_REFLECTING_SHORT, "en")
        assert result == "Reflecting..."

    def test_interpolates_variables(self) -> None:
        """Should interpolate variables into message template."""
        result = get_message(CHAT_CHATTING_WITH, "en", author="Voltaire")
        assert result == "You are now chatting with Voltaire."

    def test_interpolates_variables_french(self) -> None:
        """Should interpolate variables into French message template."""
        result = get_message(CHAT_CHATTING_WITH, "fr", author="Voltaire")
        assert result == "Vous discutez maintenant avec Voltaire."

    def test_interpolates_multiple_variables(self) -> None:
        """Should interpolate multiple variables."""
        result = get_message(ERROR_GENERATING_RESPONSE, "en", error="Connection timeout")
        assert result == "Error generating response: Connection timeout"

    def test_raises_for_missing_key(self) -> None:
        """Should raise KeyError for non-existent message key."""
        with pytest.raises(KeyError):
            get_message("nonexistent.key", "en")

    def test_raises_for_invalid_nested_key(self) -> None:
        """Should raise KeyError for invalid nested key path."""
        with pytest.raises(KeyError):
            get_message("chat.nonexistent.deeply.nested", "en")


class TestMessageKeys:
    """Tests verifying all message keys exist in locale files."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    # List of all message keys that should exist in both locales
    MESSAGE_KEYS = [
        CHAT_WELCOME,
        CHAT_CHATTING_WITH,
        CHAT_EXIT_INSTRUCTIONS,
        CHAT_INPUT_PROMPT,
        STATUS_REFLECTING_SHORT,
        STATUS_REFLECTING_VERBOSE,
        SOURCES_NONE,
        SOURCES_LABEL_CLI,
        SOURCES_LABEL_WEB,
        SOURCES_ITEM_PREFIX_CLI,
        SOURCES_ITEM_SEPARATOR_WEB,
        SOURCES_SUFFIX_WEB,
        ERROR_CHAIN_NOT_INITIALIZED,
        ERROR_GENERATING_RESPONSE,
        ERROR_GENERIC,
    ]

    @pytest.mark.parametrize("key", MESSAGE_KEYS)
    def test_key_exists_in_english(self, key: str) -> None:
        """Should find all defined message keys in English locale."""
        # Should not raise
        result = get_message(key, "en")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize("key", MESSAGE_KEYS)
    def test_key_exists_in_french(self, key: str) -> None:
        """Should find all defined message keys in French locale."""
        # Should not raise
        result = get_message(key, "fr")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_english_and_french_have_same_structure(self) -> None:
        """Should have identical key structure in both locale files."""
        en_messages = load_messages("en")
        fr_messages = load_messages("fr")

        # Should have same top-level keys
        assert set(en_messages.keys()) == set(fr_messages.keys())

        # Check each category has same sub-keys
        for category in en_messages.keys():
            if isinstance(en_messages[category], dict) and isinstance(fr_messages[category], dict):
                assert set(en_messages[category].keys()) == set(fr_messages[category].keys()), \
                    f"Mismatch in '{category}' keys between en.yaml and fr.yaml"


class TestLocaleFileIntegrity:
    """Tests verifying locale YAML files are well-formed."""

    def test_english_locale_is_valid_yaml(self) -> None:
        """Should parse English locale file without errors."""
        project_root = Path(__file__).parent.parent.parent.parent
        en_path = project_root / "locales" / "en.yaml"

        with en_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert isinstance(data, dict)

    def test_french_locale_is_valid_yaml(self) -> None:
        """Should parse French locale file without errors."""
        project_root = Path(__file__).parent.parent.parent.parent
        fr_path = project_root / "locales" / "fr.yaml"

        with fr_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert isinstance(data, dict)

    def test_all_values_are_strings(self) -> None:
        """Should verify all leaf values in locale files are strings."""
        def check_all_strings(data: dict, path: str = "") -> None:
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, dict):
                    check_all_strings(value, current_path)
                else:
                    assert isinstance(value, str), \
                        f"Non-string value at '{current_path}': {type(value)}"

        en_messages = load_messages("en")
        fr_messages = load_messages("fr")

        check_all_strings(en_messages)
        check_all_strings(fr_messages)

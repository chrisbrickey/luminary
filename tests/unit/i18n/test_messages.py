"""Unit tests for i18n message loading and retrieval."""

from unittest.mock import patch

import pytest

from src.configs.common import DEFAULT_RESPONSE_LANGUAGE, ENGLISH_ISO_CODE, FRENCH_ISO_CODE
from src.i18n import clear_cache, get_message, load_messages
from src.i18n.messages import LOCALIZATION_LANGUAGES, _get_locale_path
from src.i18n.key_registry import ALL_REQUIRED_KEYS

from src.i18n.keys import (
    CHAT_CHATTING_WITH,
    CHAT_EXIT_INSTRUCTIONS,
    CHAT_INPUT_PROMPT,
    ERROR_CHAIN_NOT_INITIALIZED,
    ERROR_GENERATING_RESPONSE,
    ERROR_GENERIC,
    SOURCES_LABEL,
    SOURCES_NONE,
    STATUS_REFLECTING,
)

class TestDefaultLanguageLocalization:
    """Tests to verify default language is always supported with
    localization of string literals in the UI via a parseable locale file.

    Some of these tests intentionally duplicate the tests on LOCALIZATION_LANGUAGES.
    Localization support for the default response language is critical to the performance of this app.
    So test coverage remains independent from implementation of LOCALIZATION_LANGUAGES."""

    def test_default_language_has_locale_file(self) -> None:
        from src.i18n.messages import _get_locale_path

        locale_path = _get_locale_path(DEFAULT_RESPONSE_LANGUAGE)
        assert locale_path.exists(), (
            f"Locale file for default language ({DEFAULT_RESPONSE_LANGUAGE}) does not exist at: {locale_path}."
        )

    def test_default_language_locale_file_loads_and_contains_required_keys(self) -> None:
        # Should not raise any exception
        messages = load_messages(DEFAULT_RESPONSE_LANGUAGE)
        assert isinstance(messages, dict)
        assert len(messages) > 0

        # Verify all required keys are present
        for key in ALL_REQUIRED_KEYS:
            result = get_message(key, DEFAULT_RESPONSE_LANGUAGE)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_default_language_listed_as_localization_language(self) -> None:
        assert DEFAULT_RESPONSE_LANGUAGE in LOCALIZATION_LANGUAGES, (
            f"Default language ({DEFAULT_RESPONSE_LANGUAGE}) is not in LOCALIZATION_LANGUAGES: {LOCALIZATION_LANGUAGES}"
        )

class TestLocalizationLanguagesConfiguration:
    """Tests to verify list of LOCALIZATION_LANGUAGES matches reality
    and that each of those languages has a valid locale file."""

    def test_all_localization_languages_have_locale_files(self) -> None:
        for language in LOCALIZATION_LANGUAGES:
            locale_path = _get_locale_path(language)
            assert locale_path.exists(), (
                f"Language ({language}) is in LOCALIZATION_LANGUAGES but no file exists at {locale_path}"
            )

    def test_no_orphaned_locale_files(self) -> None:
        """Should warn if YAML files exist that aren't in LOCALIZATION_LANGUAGES."""

        # Find all YAML files
        from src.i18n.messages import _get_locale_path
        locales_dir = _get_locale_path(DEFAULT_RESPONSE_LANGUAGE).parent
        locale_files = set(p.stem for p in locales_dir.glob("*.yaml"))

        # Isolate orphan files (not in LOCALIZATION_LANGUAGES)
        orphans = locale_files - LOCALIZATION_LANGUAGES

        # Verify no orphan files
        assert not orphans, (
            f"Found locale files not declared in LOCALIZATION_LANGUAGES: {orphans}. "
            f"Either add them to LOCALIZATION_LANGUAGES or remove the orphaned files."
        )

    @pytest.mark.parametrize("key", list(ALL_REQUIRED_KEYS))
    def test_all_localization_languages_define_all_required_keys(self, key: str) -> None:
        """All localization languages should define all required keys."""
        for language in LOCALIZATION_LANGUAGES:
            message = get_message(key, language)
            assert isinstance(message, str)
            assert len(message) > 0

    def test_french_file_has_french_language_patterns(self) -> None:
        """French locale file should contain French language patterns.

        I am explicitly testing some content of the french locale yaml file here
        because the French Enlightement is the focus of this application.

        The content of the default english yaml file is sufficiently tested in TestGetMessage.
        I do not believe it is necessary to run this kind of test for every
        language file as the size of LOCALIZATION_LANGUAGES grows."""

        # Load the french locale file
        messages = load_messages(FRENCH_ISO_CODE)

        # Collect all string values
        all_text = []
        for section in messages.values():
            if isinstance(section, dict):
                all_text.extend(v for v in section.values() if isinstance(v, str))

        # Transform all to lowercase
        combined_text = " ".join(all_text).lower()

        # Check for common French words/patterns
        # NB: I am not using a library for language detection for this test because the string
        # values are often very short - not appropriate use cases or real language detection.
        french_indicators = ["erreur", "vous", "tapez", "pas", "est", "de", "la", "avec", "sur", "dans", "pour"]
        found_indicators = sum(1 for word in french_indicators if word in combined_text)

        assert found_indicators >= 3, f"Expected French language patterns but only found {found_indicators}"

class TestLoadMessages:
    """Tests for load_messages function including correctness, caching, and default behavior."""

    def test_loads_correct_locale_file(self) -> None:
        """Should load the correct locale file."""
        
        # Setup
        loaded_file_path = None
        def capture_file_path(file_obj):
            nonlocal loaded_file_path
            loaded_file_path = file_obj.name
            # Return minimal valid YAML
            return {"chat": {"input_prompt": "Vous : "}}

        with patch("yaml.safe_load", side_effect=capture_file_path):
            load_messages(FRENCH_ISO_CODE)

            # Verify that the correct file was loaded
            assert loaded_file_path is not None, "yaml.safe_load should have been called"
            assert loaded_file_path.endswith("fr.yaml"), (
                f"Expected fr.yaml to be loaded, but got {loaded_file_path}"
            )

    def test_caches_loaded_messages(self) -> None:
        """Should cache messages to avoid re-reading files."""
        messages1 = load_messages(ENGLISH_ISO_CODE)
        messages2 = load_messages(ENGLISH_ISO_CODE)

        # Verify returns the same cached object
        assert messages1 is messages2

    def test_clear_cache_reloads_messages(self) -> None:
        """Should reload messages after cache is cleared."""
        messages1 = load_messages(ENGLISH_ISO_CODE)
        clear_cache()
        messages2 = load_messages(ENGLISH_ISO_CODE)

        # Verify not the same object even though content is the same
        assert messages1 == messages2
        assert messages1 is not messages2

    def test_falls_back_to_default_for_unlocalized_language(self) -> None:
        """Should fall back to default language when detected language
        is not supported with localization."""
        # Call for Icelandic language, which is not supported by localization
        messages = load_messages("is")

        # Verify fallback to default language (English)
        assert messages == load_messages(DEFAULT_RESPONSE_LANGUAGE)

    def test_caches_fallback_to_default_for_unlocalized_language(self) -> None:
        """Should cache the default language when falling back."""
        # First call to rare, unlocalized language (Hungarian)
        messages1 = load_messages("hu")

        # Second call to same unlocalized language
        messages2 = load_messages("hu")

        # Verify returns the cached object, which should be in default language (English)
        assert messages1 is messages2
        assert messages1 == load_messages(ENGLISH_ISO_CODE)

class TestGetMessage:
    """Tests behavior of get_message function including parsed values.

    These tests verify actual YAML content of the default language,
    boosting confidence of both the default content and the end-to-end 
    functionality of the i18n system."""

    def test_capitalizes_interpolated_author(self) -> None:
        """Should capitalize author correctly."""
        result1 = get_message(CHAT_CHATTING_WITH, DEFAULT_RESPONSE_LANGUAGE, author="voltaire")
        assert result1 == "You are now chatting with Voltaire."

        result2 = get_message(CHAT_CHATTING_WITH, DEFAULT_RESPONSE_LANGUAGE, author="VOLTAIRE")
        assert result2 == "You are now chatting with Voltaire."

    def test_raises_for_missing_key(self) -> None:
        """Should raise KeyError for non-existent message key."""
        with pytest.raises(KeyError):
            get_message("nonexistent.key", DEFAULT_RESPONSE_LANGUAGE)

    def test_raises_for_invalid_key_path(self) -> None:
        """Should raise KeyError for invalid key path."""
        with pytest.raises(KeyError, match="Invalid key path"):
            get_message("chat.chatting_with.extra", DEFAULT_RESPONSE_LANGUAGE)

    def test_parses_default_language(self) -> None:
        result1 = get_message(CHAT_INPUT_PROMPT, DEFAULT_RESPONSE_LANGUAGE)
        assert result1 == "You: "

        result2 = get_message(CHAT_EXIT_INSTRUCTIONS, DEFAULT_RESPONSE_LANGUAGE)
        assert result2 == "Type 'quit' or press Ctrl+C to exit."

        result3 = get_message(SOURCES_NONE, DEFAULT_RESPONSE_LANGUAGE)
        assert result3 == "none"

        result4 = get_message(SOURCES_LABEL, DEFAULT_RESPONSE_LANGUAGE)
        assert result4 == "**Sources:**"

        result5 = get_message(STATUS_REFLECTING, DEFAULT_RESPONSE_LANGUAGE)
        assert result5 == "Reflecting... (response time varies with the amount of data retrieved and the connection)"

    def test_parses_non_default_language(self) -> None:
        result = get_message(STATUS_REFLECTING, FRENCH_ISO_CODE)
        assert result == "Réflexion... (le délai dépend de la taille des données et la connexion)"

    def test_interpolates_variables(self) -> None:
        result = get_message(CHAT_CHATTING_WITH, DEFAULT_RESPONSE_LANGUAGE, author="Voltaire")
        assert result == "You are now chatting with Voltaire."

    def test_parses_simple_error(self) -> None:
        result = get_message(ERROR_CHAIN_NOT_INITIALIZED, DEFAULT_RESPONSE_LANGUAGE)
        assert result == "Cannot send message: chain is not initialized. Please check the configuration and error messages above."

    def test_interpolates_error_details(self) -> None:
        result1 = get_message(ERROR_GENERATING_RESPONSE, DEFAULT_RESPONSE_LANGUAGE, error="API timeout")
        assert result1 == "Error generating response: API timeout"

        result2 = get_message(ERROR_GENERIC, DEFAULT_RESPONSE_LANGUAGE, error="Connection timeout")
        assert result2 == "Error: Connection timeout"

"""Unit tests for scripts/ingest.py."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.configs.common import RAW_DATA_PATH
from src.configs.loader_configs import INGEST_CONFIGS

# Test constants
TEST_AUTHOR = "voltaire"
INVALID_AUTHOR = "nonexistent_author"

# Test data for helper function tests (not testing defaults)
TEST_RAW_PATH = "test/raw"
TEST_DB_PATH = "test/db"


class TestIngestMain:
    """Test main() function of ingest script."""

    @patch("scripts.ingest.check_ollama_or_exit")
    @patch("scripts.ingest.ingest_author")
    def test_default_arguments(
        self,
        mock_ingest: MagicMock,
        mock_ollama: MagicMock,
    ) -> None:
        """Test main() with default arguments."""
        mock_ollama.return_value = None

        with patch("sys.argv", ["ingest.py"]):
            from scripts.ingest import main
            main()

        # Verify Ollama check was called
        mock_ollama.assert_called_once()

        # Verify ingest_author called for all authors (since default author is None)
        assert mock_ingest.call_count == len(INGEST_CONFIGS)

        # Verify default directories used
        call_kwargs = mock_ingest.call_args[1]
        assert call_kwargs["raw_data_path"] == str(RAW_DATA_PATH)
        assert call_kwargs["skip_scrape"] is False
        assert call_kwargs["skip_embed"] is False

    @patch("scripts.ingest.check_ollama_or_exit")
    @patch("scripts.ingest.subprocess.run")
    def test_default_scrape_and_embed(
        self,
        mock_run: MagicMock,
        mock_ollama: MagicMock,
    ) -> None:
        """Test default behavior: both scrape and embed."""
        mock_ollama.return_value = None
        mock_run.return_value = MagicMock(returncode=0)

        with patch("sys.argv", ["ingest.py", "--author", TEST_AUTHOR]):
            from scripts.ingest import main
            main()

        # Verify Ollama check was called
        mock_ollama.assert_called_once()

        # Verify both scripts were called
        assert mock_run.call_count == 2

        # Verify scrape script called with default paths
        scrape_call = mock_run.call_args_list[0]
        assert scrape_call[0][0] == [
            "uv", "run", "python", "scripts/scrape_wikisource.py",
            "--author", TEST_AUTHOR,
            "--output-path", str(RAW_DATA_PATH)
        ]
        assert scrape_call[1]["check"] is True

        # Verify embed script called with default paths (no --db flag)
        embed_call = mock_run.call_args_list[1]
        assert embed_call[0][0] == [
            "uv", "run", "python", "scripts/embed_and_store.py",
            "--author", TEST_AUTHOR,
            "--input-path", str(RAW_DATA_PATH)
        ]
        assert embed_call[1]["check"] is True

    @patch("scripts.ingest.check_ollama_or_exit")
    @patch("scripts.ingest.subprocess.run")
    def test_skip_scrape(
        self,
        mock_run: MagicMock,
        mock_ollama: MagicMock,
    ) -> None:
        """Test --skip-scrape flag skips scraping phase."""
        mock_ollama.return_value = None
        mock_run.return_value = MagicMock(returncode=0)

        with patch("sys.argv", ["ingest.py", "--author", TEST_AUTHOR, "--skip-scrape"]):
            from scripts.ingest import main
            main()

        # Verify Ollama check was called (embed needs it)
        mock_ollama.assert_called_once()

        # Verify only embed script was called with default paths (no --db flag)
        mock_run.assert_called_once()
        embed_call = mock_run.call_args_list[0]
        assert embed_call[0][0] == [
            "uv", "run", "python", "scripts/embed_and_store.py",
            "--author", TEST_AUTHOR,
            "--input-path", str(RAW_DATA_PATH)
        ]

    @patch("scripts.ingest.subprocess.run")
    def test_skip_embed(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test --skip-embed flag skips embedding phase and Ollama check."""
        mock_run.return_value = MagicMock(returncode=0)

        with patch("sys.argv", ["ingest.py", "--author", TEST_AUTHOR, "--skip-embed"]):
            from scripts.ingest import main
            main()

        # Verify only scrape script was called with default paths
        mock_run.assert_called_once()
        scrape_call = mock_run.call_args_list[0]
        assert scrape_call[0][0] == [
            "uv", "run", "python", "scripts/scrape_wikisource.py",
            "--author", TEST_AUTHOR,
            "--output-path", str(RAW_DATA_PATH)
        ]

    @patch("scripts.ingest.check_ollama_or_exit")
    @patch("scripts.ingest.subprocess.run")
    def test_all_authors_default(
        self,
        mock_run: MagicMock,
        mock_ollama: MagicMock,
    ) -> None:
        """Test that no --author flag processes all configured authors."""
        mock_ollama.return_value = None
        mock_run.return_value = MagicMock(returncode=0)

        with patch("sys.argv", ["ingest.py"]):
            from scripts.ingest import main
            main()

        # Verify Ollama check was called once
        mock_ollama.assert_called_once()

        # Should call scripts twice per author (scrape + embed)
        assert mock_run.call_count == len(INGEST_CONFIGS) * 2

    @patch("scripts.ingest.check_ollama_or_exit")
    @patch("scripts.ingest.subprocess.run")
    def test_single_author(
        self,
        mock_run: MagicMock,
        mock_ollama: MagicMock,
    ) -> None:
        """Test processing a single author."""
        mock_ollama.return_value = None
        mock_run.return_value = MagicMock(returncode=0)

        with patch("sys.argv", ["ingest.py", "--author", TEST_AUTHOR]):
            from scripts.ingest import main
            main()

        # Verify Ollama check was called
        mock_ollama.assert_called_once()

        # Should call both scripts once
        assert mock_run.call_count == 2

    @patch("scripts.ingest.check_ollama_or_exit")
    def test_invalid_author_exits_with_error(
        self,
        mock_ollama: MagicMock,
    ) -> None:
        """Test that invalid author exits with error."""
        mock_ollama.return_value = None

        with patch("sys.argv", ["ingest.py", "--author", INVALID_AUTHOR]):
            from scripts.ingest import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_both_skip_flags_exits_with_error(self) -> None:
        """Test that using both --skip-scrape and --skip-embed exits with error."""
        with patch("sys.argv", ["ingest.py", "--author", TEST_AUTHOR, "--skip-scrape", "--skip-embed"]):
            from scripts.ingest import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("scripts.ingest.check_ollama_or_exit")
    @patch("scripts.ingest.subprocess.run")
    def test_custom_raw_data_path(
        self,
        mock_run: MagicMock,
        mock_ollama: MagicMock,
    ) -> None:
        """Test using custom raw-dir."""
        custom_raw = "custom/raw"
        mock_ollama.return_value = None
        mock_run.return_value = MagicMock(returncode=0)

        with patch("sys.argv", [
            "ingest.py",
            "--author", TEST_AUTHOR,
            "--raw-data-path", custom_raw
        ]):
            from scripts.ingest import main
            main()

        # Verify custom paths used in both scripts
        scrape_call = mock_run.call_args_list[0]
        assert scrape_call[0][0] == [
            "uv", "run", "python", "scripts/scrape_wikisource.py",
            "--author", TEST_AUTHOR,
            "--output-path", custom_raw
        ]

        embed_call = mock_run.call_args_list[1]
        assert embed_call[0][0] == [
            "uv", "run", "python", "scripts/embed_and_store.py",
            "--author", TEST_AUTHOR,
            "--input-path", custom_raw
        ]

    @patch("scripts.ingest.check_ollama_or_exit")
    def test_ollama_not_available_exits_with_error(
        self,
        mock_ollama: MagicMock,
    ) -> None:
        """Test that Ollama not available exits with error (when embedding)."""
        # check_ollama_or_exit handles the error internally and calls sys.exit(1)
        mock_ollama.side_effect = SystemExit(1)

        with patch("sys.argv", ["ingest.py", "--author", TEST_AUTHOR]):
            from scripts.ingest import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("scripts.ingest.subprocess.run")
    def test_ollama_not_checked_when_skip_embed(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test that Ollama is not checked when --skip-embed is used."""
        mock_run.return_value = MagicMock(returncode=0)

        with patch("scripts.ingest.check_ollama_or_exit") as mock_ollama:
            with patch("sys.argv", ["ingest.py", "--author", TEST_AUTHOR, "--skip-embed"]):
                from scripts.ingest import main
                main()

            # Ollama check should not be called
            mock_ollama.assert_not_called()

    @patch("scripts.ingest.check_ollama_or_exit")
    @patch("scripts.ingest.subprocess.run")
    def test_subprocess_error_exits_with_error(
        self,
        mock_run: MagicMock,
        mock_ollama: MagicMock,
    ) -> None:
        """Test that subprocess.CalledProcessError exits with error."""
        mock_ollama.return_value = None
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")

        with patch("sys.argv", ["ingest.py", "--author", TEST_AUTHOR]):
            from scripts.ingest import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("scripts.ingest.check_ollama_or_exit")
    @patch("scripts.ingest.subprocess.run")
    def test_general_exception_exits_with_error(
        self,
        mock_run: MagicMock,
        mock_ollama: MagicMock,
    ) -> None:
        """Test that general exception exits with error."""
        mock_ollama.return_value = None
        mock_run.side_effect = Exception("Unexpected error")

        with patch("sys.argv", ["ingest.py", "--author", TEST_AUTHOR]):
            from scripts.ingest import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestIngestAuthor:
    """Test ingest_author() helper function."""

    @patch("scripts.ingest.subprocess.run")
    def test_both_phases_executed(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test that both phases are executed when no skip flags."""
        mock_run.return_value = MagicMock(returncode=0)

        from scripts.ingest import ingest_author
        mock_logger = MagicMock()
        ingest_author(
            logger=mock_logger,
            author=TEST_AUTHOR,
            raw_data_path=TEST_RAW_PATH,
            skip_scrape=False,
            skip_embed=False
        )

        # Both scripts should be called
        assert mock_run.call_count == 2

    @patch("scripts.ingest.subprocess.run")
    def test_skip_scrape_phase(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test skipping scrape phase."""
        mock_run.return_value = MagicMock(returncode=0)

        from scripts.ingest import ingest_author
        mock_logger = MagicMock()
        ingest_author(
            logger=mock_logger,
            author=TEST_AUTHOR,
            raw_data_path=TEST_RAW_PATH,
            skip_scrape=True,
            skip_embed=False
        )

        # Only embed script should be called
        mock_run.assert_called_once()
        embed_call = mock_run.call_args_list[0]
        assert "embed_and_store.py" in embed_call[0][0][3]

    @patch("scripts.ingest.subprocess.run")
    def test_skip_embed_phase(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test skipping embed phase."""
        mock_run.return_value = MagicMock(returncode=0)

        from scripts.ingest import ingest_author
        mock_logger = MagicMock()
        ingest_author(
            logger=mock_logger,
            author=TEST_AUTHOR,
            raw_data_path=TEST_RAW_PATH,
            skip_scrape=False,
            skip_embed=True
        )

        # Only scrape script should be called
        mock_run.assert_called_once()
        scrape_call = mock_run.call_args_list[0]
        assert "scrape_wikisource.py" in scrape_call[0][0][3]

    def test_invalid_author_raises_value_error(self) -> None:
        """Test that invalid author raises ValueError."""
        from scripts.ingest import ingest_author

        mock_logger = MagicMock()
        with pytest.raises(ValueError, match="Unknown author"):
            ingest_author(
                logger=mock_logger,
                author=INVALID_AUTHOR,
                raw_data_path=TEST_RAW_PATH,
                skip_scrape=False,
                skip_embed=False
            )

    @patch("scripts.ingest.subprocess.run")
    def test_custom_directory_passed_through(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test that custom directory is passed to scripts."""
        custom_raw = "custom/raw"
        mock_run.return_value = MagicMock(returncode=0)

        from scripts.ingest import ingest_author
        mock_logger = MagicMock()
        ingest_author(
            logger=mock_logger,
            author=TEST_AUTHOR,
            raw_data_path=custom_raw,
            skip_scrape=False,
            skip_embed=False
        )

        # Verify custom paths in both calls
        scrape_call = mock_run.call_args_list[0]
        assert "--output-path" in scrape_call[0][0]
        assert custom_raw in scrape_call[0][0]

        embed_call = mock_run.call_args_list[1]
        assert "--input-path" in embed_call[0][0]
        assert custom_raw in embed_call[0][0]

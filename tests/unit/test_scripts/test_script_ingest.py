"""Unit tests for scripts/ingest.py."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Test constants
TEST_AUTHOR = "voltaire"
INVALID_AUTHOR = "nonexistent_author"
TEST_RAW_DIR = "data/raw"
TEST_DB_PATH = "data/chroma_db"
ALL_AUTHORS = ["voltaire"]  # List of all authors in INGEST_CONFIGS


class TestIngestMain:
    """Test main() function of ingest script."""

    @patch("scripts.ingest.check_ollama_available")
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

        # Verify scrape script called
        scrape_call = mock_run.call_args_list[0]
        assert scrape_call[0][0] == [
            "uv", "run", "python", "scripts/scrape_wikisource.py",
            "--author", TEST_AUTHOR,
            "--output-dir", TEST_RAW_DIR
        ]
        assert scrape_call[1]["check"] is True

        # Verify embed script called
        embed_call = mock_run.call_args_list[1]
        assert embed_call[0][0] == [
            "uv", "run", "python", "scripts/embed_and_store.py",
            "--author", TEST_AUTHOR,
            "--input-dir", TEST_RAW_DIR,
            "--db", TEST_DB_PATH
        ]
        assert embed_call[1]["check"] is True

    @patch("scripts.ingest.check_ollama_available")
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

        # Verify only embed script was called
        mock_run.assert_called_once()
        embed_call = mock_run.call_args_list[0]
        assert embed_call[0][0] == [
            "uv", "run", "python", "scripts/embed_and_store.py",
            "--author", TEST_AUTHOR,
            "--input-dir", TEST_RAW_DIR,
            "--db", TEST_DB_PATH
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

        # Verify only scrape script was called
        mock_run.assert_called_once()
        scrape_call = mock_run.call_args_list[0]
        assert scrape_call[0][0] == [
            "uv", "run", "python", "scripts/scrape_wikisource.py",
            "--author", TEST_AUTHOR,
            "--output-dir", TEST_RAW_DIR
        ]

    @patch("scripts.ingest.check_ollama_available")
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
        assert mock_run.call_count == len(ALL_AUTHORS) * 2

    @patch("scripts.ingest.check_ollama_available")
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

    @patch("scripts.ingest.check_ollama_available")
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

    @patch("scripts.ingest.check_ollama_available")
    @patch("scripts.ingest.subprocess.run")
    def test_custom_directories(
        self,
        mock_run: MagicMock,
        mock_ollama: MagicMock,
    ) -> None:
        """Test using custom raw-dir and db paths."""
        custom_raw = "custom/raw"
        custom_db = "custom/db"
        mock_ollama.return_value = None
        mock_run.return_value = MagicMock(returncode=0)

        with patch("sys.argv", [
            "ingest.py",
            "--author", TEST_AUTHOR,
            "--raw-dir", custom_raw,
            "--db", custom_db
        ]):
            from scripts.ingest import main
            main()

        # Verify custom paths used in both scripts
        scrape_call = mock_run.call_args_list[0]
        assert scrape_call[0][0] == [
            "uv", "run", "python", "scripts/scrape_wikisource.py",
            "--author", TEST_AUTHOR,
            "--output-dir", custom_raw
        ]

        embed_call = mock_run.call_args_list[1]
        assert embed_call[0][0] == [
            "uv", "run", "python", "scripts/embed_and_store.py",
            "--author", TEST_AUTHOR,
            "--input-dir", custom_raw,
            "--db", custom_db
        ]

    @patch("scripts.ingest.check_ollama_available")
    def test_ollama_not_available_exits_with_error(
        self,
        mock_ollama: MagicMock,
    ) -> None:
        """Test that Ollama not available exits with error (when embedding)."""
        mock_ollama.side_effect = RuntimeError("Ollama is not running")

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

        with patch("scripts.ingest.check_ollama_available") as mock_ollama:
            with patch("sys.argv", ["ingest.py", "--author", TEST_AUTHOR, "--skip-embed"]):
                from scripts.ingest import main
                main()

            # Ollama check should not be called
            mock_ollama.assert_not_called()

    @patch("scripts.ingest.check_ollama_available")
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

    @patch("scripts.ingest.check_ollama_available")
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
        ingest_author(
            author=TEST_AUTHOR,
            raw_dir=TEST_RAW_DIR,
            db_dir=TEST_DB_PATH,
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
        ingest_author(
            author=TEST_AUTHOR,
            raw_dir=TEST_RAW_DIR,
            db_dir=TEST_DB_PATH,
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
        ingest_author(
            author=TEST_AUTHOR,
            raw_dir=TEST_RAW_DIR,
            db_dir=TEST_DB_PATH,
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

        with pytest.raises(ValueError, match="Unknown author"):
            ingest_author(
                author=INVALID_AUTHOR,
                raw_dir=TEST_RAW_DIR,
                db_dir=TEST_DB_PATH,
                skip_scrape=False,
                skip_embed=False
            )

    @patch("scripts.ingest.subprocess.run")
    def test_custom_directories_passed_through(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test that custom directories are passed to scripts."""
        custom_raw = "custom/raw"
        custom_db = "custom/db"
        mock_run.return_value = MagicMock(returncode=0)

        from scripts.ingest import ingest_author
        ingest_author(
            author=TEST_AUTHOR,
            raw_dir=custom_raw,
            db_dir=custom_db,
            skip_scrape=False,
            skip_embed=False
        )

        # Verify custom paths in both calls
        scrape_call = mock_run.call_args_list[0]
        assert "--output-dir" in scrape_call[0][0]
        assert custom_raw in scrape_call[0][0]

        embed_call = mock_run.call_args_list[1]
        assert "--input-dir" in embed_call[0][0]
        assert custom_raw in embed_call[0][0]
        assert "--db" in embed_call[0][0]
        assert custom_db in embed_call[0][0]

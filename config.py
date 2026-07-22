from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
RAW_DATA_DIR = DATA_DIR / "raw"

CONSTITUENTS_FILE = METADATA_DIR / "sp500_constituents.csv"
DOWNLOAD_LOG_FILE = METADATA_DIR / "download_log.csv"

SECURITY_QUALITY_FILE = METADATA_DIR / "security_quality.csv"
QUALITY_SUMMARY_FILE = METADATA_DIR / "quality_summary.csv"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATA_OVERRIDES_FILE = (
    METADATA_DIR / "data_overrides.csv"
)

CLEANING_LOG_FILE = (
    METADATA_DIR / "cleaning_log.csv"
)

PANEL_DATA_DIR = DATA_DIR / "panels"

OPEN_PANEL_FILE = PANEL_DATA_DIR / "open.parquet"
CLOSE_PANEL_FILE = PANEL_DATA_DIR / "close.parquet"
VOLUME_PANEL_FILE = PANEL_DATA_DIR / "volume.parquet"
DOLLAR_VOLUME_PANEL_FILE = (
    PANEL_DATA_DIR / "median_dollar_volume_20d.parquet"
)
ELIGIBILITY_PANEL_FILE = (
    PANEL_DATA_DIR / "eligibility.parquet"
)
UNIVERSE_COVERAGE_FILE = (
    METADATA_DIR / "universe_coverage.csv"
)

CANDIDATE_PAIRS_FILE = (
    DATA_DIR / "research" / "candidate_pairs.parquet"
)

CANDIDATE_SUMMARY_FILE = (
    METADATA_DIR / "candidate_pair_summary.csv"
)

RESEARCH_DATA_DIR = DATA_DIR / "research"

COINTEGRATION_RESULTS_FILE = (
    RESEARCH_DATA_DIR / "cointegration_results.parquet"
)

QUALIFIED_PAIRS_FILE = (
    RESEARCH_DATA_DIR / "qualified_pairs.parquet"
)

COINTEGRATION_SUMMARY_FILE = (
    METADATA_DIR / "cointegration_summary.csv"
)

MAX_CANDIDATES_PER_SECTOR = 50

COINTEGRATION_MAX_LAG = 1
FDR_THRESHOLD = 0.10

MINIMUM_HALF_LIFE = 2.0
MAXIMUM_HALF_LIFE = 60.0

MINIMUM_HEDGE_RATIO = 0.10
MAXIMUM_HEDGE_RATIO = 10.0

MINIMUM_SPREAD_STANDARD_DEVIATION = 0.001


# ============================================================
# DATA PARAMETERS
# ============================================================

START_DATE = "2000-01-01"

# None downloads data up to the latest available trading day.
END_DATE = None

# Download several securities in each Yahoo request.
BATCH_SIZE = 25

# Failed securities will be attempted individually.
MAX_RETRIES = 3

# Pause between individual retry attempts.
RETRY_WAIT_SECONDS = 5

MINIMUM_HISTORY = 252
MINIMUM_PRICE = 5.0

LIQUIDITY_WINDOW = 20
MINIMUM_LIQUID_DAYS = 15
MINIMUM_MEDIAN_DOLLAR_VOLUME = 10_000_000

FORMATION_WINDOW = 252
MINIMUM_PAIR_OBSERVATIONS = 200
MINIMUM_RETURN_CORRELATION = 0.50
MAX_PARTNERS_PER_STOCK = 10

# The last available month may be incomplete.
INCLUDE_PARTIAL_FINAL_MONTH = False

BASELINE_CANDIDATE_CAP = 20

BASELINE_PAIRS_FILE = (
    RESEARCH_DATA_DIR / "baseline_pairs.parquet"
)

BASELINE_PAIR_SUMMARY_FILE = (
    METADATA_DIR / "baseline_pair_summary.csv"
)

SPREAD_EVENT_FILE = (
    RESEARCH_DATA_DIR / "spread_events.parquet"
)

SPREAD_EVENT_SUMMARY_FILE = (
    METADATA_DIR / "spread_event_summary.csv"
)

SPREAD_EVENT_SECTOR_FILE = (
    METADATA_DIR / "spread_event_by_sector.csv"
)

ENTRY_Z_SCORE = 2.0
PARTIAL_EXIT_Z_SCORE = 1.0
STANDARD_EXIT_Z_SCORE = 0.5

# ============================================================
# DIRECTORY SETUP
# ============================================================

for directory in [
    DATA_DIR,
    METADATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PANEL_DATA_DIR,
    RESEARCH_DATA_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
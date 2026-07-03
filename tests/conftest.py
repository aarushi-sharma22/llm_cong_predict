import sys
from pathlib import Path

# Make the tests/ directory importable so `fixtures.synthetic_data` resolves.
sys.path.insert(0, str(Path(__file__).parent))

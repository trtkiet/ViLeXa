import sys
from pathlib import Path

# Ensure project root is on sys.path so 'backend' package can be imported during tests
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

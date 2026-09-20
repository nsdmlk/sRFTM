import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import SRFTM
from src.math_tokenizer import MathTokenizer
from src.templates import build_template

# Перебиваем __module__ чтобы pickle не тянул src.*
SRFTM.__module__ = 'srftm_models'
MathTokenizer.__module__ = 'srftm_models'
build_template.__module__ = 'srftm_models'
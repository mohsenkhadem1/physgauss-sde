"""Evaluate the separately declared validation-motivated 5k refinement study."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.pidg_comparison import finish
finish.BASE=finish.ROOT/'runs/pidg-comparison-v02'
finish.NAMES=['refined-pidg','refined-ours-00001','refined-ours-0001']
if __name__=='__main__':finish.main()

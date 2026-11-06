"""Regenerate the five included legacy figures from their frozen sources."""
from pathlib import Path
import sys
import figure_style
ROOT=Path(__file__).resolve().parents[3];PAPER=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
names=['energy_block','dynamic_results','qualitative','held_visual_results','held_error_plate']
figure_style.redirect.update({n:PAPER/'figures'/('sim-'+n) for n in names})
(ROOT/'tmp/pdfs').mkdir(parents=True,exist_ok=True)
figure_style.redirect['method_overview']=ROOT/'tmp/pdfs/unused-legacy-method'
p=ROOT/'papers/cvpr2027-compositional-mechanics/generate_figures.py';source=p.read_text();source=source[:source.index("forward=read(V/")]
exec(compile(source,str(p),'exec'),{'__file__':str(p),'__name__':'__figure_regeneration__'})
sys.path.insert(0,str(ROOT/'papers/cvpr2027-compositional-mechanics/tools'))
import design_editorial_figures as d
d.results();d.plate()

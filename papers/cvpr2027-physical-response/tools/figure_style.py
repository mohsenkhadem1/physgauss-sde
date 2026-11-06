"""Consistent Arial/black typography; data colours and geometry stay unchanged."""
from pathlib import Path
from matplotlib import font_manager,rcParams
from matplotlib.figure import Figure
from matplotlib.text import Text
for p in (Path.home()/'.local/share/fonts/arial').glob('*.TTF'):font_manager.fontManager.addfont(str(p))
font_manager.findfont('Arial',fallback_to_default=False)
original=Figure.savefig
redirect={}
def save(self,path,*args,**kwargs):
 rcParams.update({'font.family':'Arial','mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','mathtext.sf':'Arial','pdf.fonttype':42})
 for text in self.findobj(match=Text):
  if text.get_text()==r'$W=e^s E_0\sum_j c_j(h,f)b_j(F)$':text.set_text('W = exp(s) E₀ Σj cj(h,f) bj(F)');text.set_fontsize(8.5)
  if text.get_text()==r'$P=\partial W/\partial F$'+'\n'+r'$M\ddot u=-\nabla_u\Pi$':text.set_text('P = ∂W/∂F\nM ü = −∇u Π')
  text.set_fontfamily('Arial');text.set_color('black');text.set_math_fontfamily('custom')
 path=Path(path);path=redirect.get(path.stem,path.with_suffix('')).with_suffix(path.suffix)
 return original(self,path,*args,**kwargs)
Figure.savefig=save

"""Evidence-grounded vector figures for the CVPR physical-interaction manuscript.

All plotted data come from sealed benchmark scores and the archived benchmark
render; no simulator, tracker, or fitting process is run here.
"""
import argparse, json
from pathlib import Path
import fitz, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE=Path(__file__).resolve().parent
BLUE="#2563A6"; TEAL="#087F72"; ORANGE="#C46C37"; INK="#162939"; MUTED="#5C6973"
YELLOW="#D5A143"; PLUM="#9D6091"; LIGHT="#EFF4F6"; GRID="#D8E1E4"
COL={"ours (RGB)":BLUE,"GIC":ORANGE,"PAC-NeRF":TEAL,"M-PhyGs":YELLOW,"PhysTwin":PLUM}
METHODS=["ours (RGB)","M-PhyGs","PhysTwin","GIC","PAC-NeRF"]

def setup():
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"font.size":9,
      "figure.facecolor":"white","axes.spines.top":False,"axes.spines.right":False,
      "axes.edgecolor":MUTED,"axes.labelcolor":INK,"text.color":INK,
      "xtick.color":MUTED,"ytick.color":MUTED})

def box(ax,xy,width,height,title,body="",color=LIGHT,border=GRID):
    x,y=xy
    ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle="round,pad=0.016,rounding_size=0.019",
                 linewidth=1,edgecolor=border,facecolor=color))
    ax.text(x+.024,y+height-.048,title,fontsize=10,fontweight="bold",ha="left",va="top")
    if body:ax.text(x+.024,y+height-.118,body,fontsize=8.4,color=MUTED,ha="left",va="top",linespacing=1.4)

def method_figure(out):
    fig,ax=plt.subplots(figsize=(13.3,4.7));fig.subplots_adjust(left=.017,right=.984,bottom=.08,top=.94)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    box(ax,(.015,.39),.23,.51,"Available evidence","RGB frames + camera calibration\nKnown metric rest geometry\nMeasured commands when available")
    box(ax,(.295,.69),.31,.22,"Known-command dynamics","RGB point tracks + Bayesian bank\nCandidate E, density, support and inclusion",color="#E9F3FB",border="#BCD3EA")
    box(ax,(.295,.37),.31,.22,"Unknown-load inverse problem","Gaussian material / support / traction fields\nAnalytic force marginalization",color="#EFF7F0",border="#C6DFCF")
    box(ax,(.655,.45),.315,.34,"Action-conditioned prediction","Roll out each physical hypothesis for a*\nPosterior response q(z,a*)\nTolerance-exceedance risk",color="#E9F0F8",border="#ABC5DF")
    box(ax,(.655,.07),.315,.23,"Decision: proceed, look or probe","Choose the least costly useful new evidence\nUpdate response distribution (separate study)",color="#F9F2E4",border="#E4C99C")
    for a,b,c in [((.245,.67),(.295,.78),MUTED),((.245,.56),(.295,.48),MUTED),((.605,.80),(.655,.67),BLUE),
                  ((.605,.47),(.655,.54),TEAL),((.812,.45),(.812,.31),YELLOW)]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=12,linewidth=1.6,color=c))
    fig.text(.02,.985,"Two inference settings, one response-oriented decision",fontsize=14,fontweight="bold",va="top")
    fig.text(.02,.017,"Appearance is evaluated on a held-out camera; the physical posterior uses tracked motion, not RGB pixel likelihoods.",fontsize=8.4,color=MUTED)
    fig.savefig(out,bbox_inches="tight");plt.close(fig)

def scores_figure(rows,out):
    panels=[("S1","rms_mm","Changed-direction force","3D error (mm)"),
            ("S2","rms_mm","Hidden-inclusion press","3D error (mm)"),
            ("S3","rms_mm","Free-impact deformation","3D error (mm)"),
            ("S3","impulse_err_Ns","Transferred impulse","Absolute error (N s)")]
    fig,axes=plt.subplots(1,4,figsize=(15.4,3.9))
    fig.subplots_adjust(left=.105,right=.98,top=.79,bottom=.24,wspace=.48)
    for ax,(sc,metric,title,unit) in zip(axes,panels):
        for pos,m in enumerate(METHODS):
            pre=[r[metric] for r in rows if r["scenario"]==sc and r["method"]==m and r["evidence"]=="passive only"]
            post=[r[metric] for r in rows if r["scenario"]==sc and r["method"]==m and r["evidence"]=="after poke"]
            if len(pre)!=3 or len(post)!=3:raise ValueError(f"Expected 3 case records for {sc}/{m}/{metric}")
            b=float(np.mean(pre));a=float(np.mean(post))
            ax.plot([b,a],[pos,pos],lw=1.5,color=COL[m],alpha=.8,zorder=1)
            ax.scatter([b],[pos],s=55,edgecolor=COL[m],facecolor="white",lw=1.7,zorder=3)
            ax.scatter([a],[pos],s=64,color=COL[m],zorder=4)
            ax.scatter(post,[pos+.17]*3,s=14,color=COL[m],alpha=.44,lw=0,zorder=5)
        ax.set_yticks(range(5));ax.set_yticklabels([x.replace("ours (RGB)","Ours (RGB)") for x in METHODS] if ax is axes[0] else [])
        ax.invert_yaxis();ax.set_ylim(4.62,-.55);ax.grid(axis="x",lw=.55,color=GRID);ax.set_axisbelow(True)
        ax.tick_params(axis="y",length=0);ax.set_title(title,loc="left",fontsize=10.2,fontweight="bold",pad=12)
        ax.set_xlabel(unit,fontsize=9)
        if metric=="impulse_err_Ns":ax.set_xscale("log");ax.set_xlim(.00025,3)
        else:ax.set_xlim(-.1,max(r[metric] for r in rows if r["scenario"]==sc and r["method"] in METHODS and r["evidence"] in ("passive only","after poke"))*1.15)
    fig.text(.105,.98,"A known calibration action changes counterfactual predictive accuracy",fontsize=13.2,fontweight="bold",va="top")
    fig.text(.105,.87,"Open circle: initial RGB evidence     Filled: after calibration     Small dots: individual test worlds",fontsize=9,color=MUTED)
    fig.savefig(out,bbox_inches="tight");plt.close(fig)

def qualitative(src,outdir):
    source=fitz.open(src)
    cuts=[("S1",(0,46,774,234),"S1  Deformable cantilever | 20 N force in a new direction"),
          ("S2",(0,266,774,477),"S2  Heterogeneous block | off-centre press above concealed stiffness"),
          ("S3",(0,497,774,691),"S3  Deformable cantilever | free striker contact and rebound")]
    for sc,rect,title in cuts:
        clip=fitz.Rect(rect);width=774;height=clip.height+49
        d=fitz.open();p=d.new_page(width=width,height=height)
        p.draw_rect(fitz.Rect(0,0,width,40),color=None,fill=(.93,.96,.98))
        p.insert_text((14,25),title,fontname="hebo",fontsize=13,color=(.08,.17,.24))
        p.show_pdf_page(fitz.Rect(0,43,width,43+clip.height),source,0,clip=clip,keep_proportion=False)
        d.save(outdir/f"qualitative-{sc}.pdf",garbage=4,deflate=True);d.close()
    source.close()

def render_figure(data,out):
    summary=data["summary_over_worlds"]
    fig,axes=plt.subplots(1,3,figsize=(12.8,3.8))
    fig.subplots_adjust(left=.08,right=.985,top=.75,bottom=.19,wspace=.29)
    for ax,sc,title in zip(axes,["S1","S2","S3"],["Changed-direction cantilever","Hidden-inclusion block","Striker collision"]):
        for m in METHODS:
            matches=[r for r in summary if r["scenario"]==sc and r["renderer"]=="shared GIC appearance" and r["method"]==m and r["stage"]=="D1"]
            if len(matches)!=1:raise ValueError(f"Expected one shared-appearance row for {sc}/{m}")
            r=matches[0];x=r["physics_rms_mm_mean"];y=r["fg_psnr_db_mean"]
            ax.scatter([x],[y],s=75,color=COL[m],edgecolor="white",lw=.7,zorder=3)
            if m in ("ours (RGB)","GIC","PAC-NeRF"):
                ax.annotate("Ours" if m=="ours (RGB)" else m,(x,y),xytext=(4,5),textcoords="offset points",fontsize=7.6,color=COL[m])
        truth=[r["fg_psnr_db_mean"] for r in summary if r["scenario"]==sc and r["renderer"]=="shared GIC appearance" and r["method"]=="true motion" and r["stage"]=="truth"]
        if truth:ax.axhline(truth[0],ls="--",lw=1,color=MUTED,alpha=.55)
        ax.set_title(title,fontsize=10,fontweight="bold",pad=10)
        ax.set_xlabel("3D motion error (mm) ↓",fontsize=9);ax.grid(color=GRID,alpha=.8,lw=.6)
        ax.set_ylim(19.85,21.08) if sc=="S2" else ax.set_ylim(17.8,22.3)
    axes[0].set_ylabel("Shared-appearance PSNR (dB) ↑",fontsize=9)
    fig.text(.08,.98,"When good rendering conceals inaccurate physics",fontweight="bold",fontsize=13.4,va="top")
    fig.text(.08,.88,"A fixed Gaussian appearance renders sealed motion fields from an unseen camera; dashed line = true motion.",color=MUTED,fontsize=9)
    fig.savefig(out,bbox_inches="tight");plt.close(fig)

def decision_figure(out):
    names=["Repeated action","Changed load\nbase visible","Changed load\nbase hidden"]
    choices=["Proceed","Look","Probe"];counts=np.array([[96,0,0],[4,92,0],[5,0,91]])
    errors=[.104,.073,.119];colors=[BLUE,TEAL,YELLOW]
    fig=plt.figure(figsize=(11.9,3.4))
    gs=fig.add_gridspec(1,2,width_ratios=[1.7,1.0],left=.06,right=.985,top=.78,bottom=.18,wspace=.31)
    ax=fig.add_subplot(gs[0,0]);offset=np.zeros(3)
    for i,(choice,c) in enumerate(zip(choices,colors)):
        ax.barh(range(3),counts[:,i],left=offset,color=c,label=choice,height=.54)
        for j,n in enumerate(counts[:,i]):
            if n:ax.text(offset[j]+n/2,j,str(n),va="center",ha="center",fontsize=10,fontweight="bold",color="white" if i<2 else INK)
        offset+=counts[:,i]
    ax.set_xlim(0,96);ax.set_xticks([0,24,48,72,96]);ax.set_xlabel("Decisions across 96 cases per condition")
    ax.set_yticks(range(3),names);ax.invert_yaxis()
    fig.legend(ax.containers[:3],choices,loc="upper right",bbox_to_anchor=(.99,.92),frameon=False,ncol=3,fontsize=9)
    ax2=fig.add_subplot(gs[0,1]);ax2.barh(range(3),errors,color=colors,height=.54)
    ax2.invert_yaxis();ax2.set_xlim(0,.16);ax2.set_yticks(range(3),[""]*3)
    ax2.set_xlabel("Final future-response RMSE (mm) ↓")
    for j,v in enumerate(errors):ax2.text(v+.006,j,f"{v:.3f}",va="center",fontsize=9.5,fontweight="bold")
    for a in (ax,ax2):a.grid(axis="x",color=GRID,lw=.6);a.set_axisbelow(True);a.tick_params(axis="y",length=0)
    fig.text(.06,.97,"Which evidence resolves the next action?",fontweight="bold",fontsize=14.5,va="top")
    fig.text(.06,.835,"Cost-aware risk policy  |  tolerance 0.50 mm  |  view cost 0.015, probe cost 0.040",fontsize=9.1,color=MUTED)
    fig.savefig(out,bbox_inches="tight");plt.close(fig)

def teaser(src,out):
    original=fitz.open(src);d=fitz.open();p=d.new_page(width=770,height=290)
    titles=["S1  NEW FORCE","S2  HIDDEN MATERIAL","S3  FREE IMPACT"]
    sub=["1.27 mm 3D RMSE","0.33 mm 3D RMSE","0.003 N s impulse error"]
    for j,(title,result,off) in enumerate(zip(titles,sub,[46,266,497])):
        x=8+j*256
        p.draw_rect(fitz.Rect(x,4,x+247,286),color=(.80,.85,.88),fill=(.985,.991,.994),width=.5)
        p.insert_text((x+11,22),title,fontname="hebo",fontsize=10.3,color=(.10,.19,.25))
        p.insert_text((x+11,40),result,fontname="hebo",fontsize=12.2,color=(.11,.41,.66))
        for lab,y,clip in [
          ("REFERENCE",51,fitz.Rect(10,off+20,169,off+80)),
          ("GIC PREDICTION",131,fitz.Rect(190,off+20,327,off+80)),
          ("OURS (RGB)",211,fitz.Rect(333,off+106,496,off+166))]:
            p.insert_text((x+10,y),lab,fontname="hebo" if lab.startswith("OURS") else "helv",fontsize=7.5,
                          color=(.11,.41,.66) if lab.startswith("OURS") else (.33,.39,.42))
            p.show_pdf_page(fitz.Rect(x+11,y+5,x+237,y+68),original,0,clip=clip,keep_proportion=False)
    d.save(out,garbage=4,deflate=True,no_new_id=True);d.close();original.close()

def main(repo,out):
    setup();out.mkdir(parents=True,exist_ok=True)
    source=repo/"output/paper-figures/fig1-benchmark.pdf"
    scores=repo/"reports/fig1-v01/fig1-scores.json"
    rendering=repo/"reports/fig1-v01/render-scores.json"
    if not source.exists():source=out/"benchmark-source.pdf"
    if not scores.exists():scores=out/"benchmark-scores.json"
    if not rendering.exists():rendering=out/"benchmark-render-scores.json"
    rows=json.loads(scores.read_text())["rows"]
    report=json.loads(rendering.read_text())
    method_figure(out/"method-overview.pdf")
    scores_figure(rows,out/"benchmark-deltas.pdf")
    qualitative(source,out)
    render_figure(report,out/"render-vs-physics.pdf")
    decision_figure(out/"evidence-decision.pdf")
    teaser(source,out/"task-teaser.pdf")
    print("Rebuilt 8 prospective benchmark and methodology visualizations")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,default=HERE.parents[2])
    parser.add_argument("--outdir",type=Path,default=HERE)
    a=parser.parse_args();main(a.repo,a.outdir)

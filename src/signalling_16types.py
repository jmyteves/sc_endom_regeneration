"""Control-anchored cell-cell signalling across the 16 cell types (extra panel, former Fig. 6).

    build(repo_root, out_dir) -> {panel_label: svg_path}   one composite sheet

  a-d  3D sender / receiver / timepoint, one per compartment
  e    four control-anchored circle plots, one per postpartum stage, two layers:
         ribbons = compartment-level change in directional preference, intensity by
                   lightness (pale = decreased, saturated = increased vs control); arc
                   span goes to the top 8 pairs by |delta|; self-pairs render as a lens
         lines   = individual cell-type pairs, thickness = |change in interaction count|,
                   grey at alpha 0.30, strongest quartile only (about 60 of 240 per panel)
  f    compartment signalling load, log2 vs control
  g    directional preference toward the epithelium per sender compartment (log2 O/E)
  h    epithelium as receiver (all cell types -> Epithelial), share of signal per stage
  i    epithelium as sender (Epithelial -> all cell types), share of signal per stage

Data is the 16-type LIANA run. load_inputs() rejects the earlier 12-type table, which
had no endothelium and mislabelled cluster 20. The composite is not in the v10 figure
set; Fig. 5 and Fig. 6 draw the epithelial ligand inputs from liana_panels.py instead.
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.colors import to_rgb, to_hex
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d

import render_svg


def _repo_root_from_here():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in [here] + [os.path.abspath(os.path.join(here, *[".."] * k)) for k in (1, 2, 3)]:
        if all(os.path.isdir(os.path.join(cand, d)) for d in ("src", "data", "scripts")):
            return cand
    raise RuntimeError("cannot locate the sc_endom_2 repo root from " + here)


# The pre-rendered share tables the original analysis wrote (LR_share_render.csv,
# LR_share_labels.csv) are not read here: the handoff records them as SATURATED, every pair
# at share 0 or 1 and every label tied at +/-0.5. h and i are computed from the full
# LR_share_vs_control.csv instead, which carries the per-pair values the saturation destroyed.
# The 25 MB liana_all_stages_16types.csv.gz is likewise never opened by a panel.
_TABLES = {
    "cmap":  "cluster_compartment_map.csv",            # node -> compartment, 16 types
    "oe":    "directional_OE_by_stage.csv",            # panel g + panel e ribbons
    "dsum":  "directional_summary_by_stage.csv",       # panel g
    "sr":    "sender_receiver_by_stage.csv",           # panels a-d
    "cload": "compartment_load_log2_vs_control.csv",   # panel f
    "edge":  "edge_counts_by_stage.csv",               # panel e lines
    "lrf":   "LR_share_vs_control.csv",                # panels h, i (full, 149,387 rows)
}

# The end labels as delivered with the shipped SVGs. Read only to assert that the ligand
# ranking recomputed below still lands on the deck's 40 labels.
_ENDLAB = "Fig6_epithelium_end_labels.csv"


def load_inputs(repo_root):
    """Read the seven tables the panels actually draw from, plus the shared palette."""
    d = os.path.join(repo_root, "data", "raw_light", "fig6")
    out = {k: pd.read_csv(os.path.join(d, f)) for k, f in _TABLES.items()}
    out["endlab"] = pd.read_csv(os.path.join(d, _ENDLAB))
    out["palj"] = json.load(open(os.path.join(repo_root, "data", "raw_light",
                                              "palette_final.json")))
    if len(out["cmap"]) != 16:
        raise ValueError(f"expected 16 cell types, got {len(out['cmap'])}")
    if "Myeloid (cl20)" in set(out["cmap"].cell_type):
        raise ValueError("'Myeloid (cl20)' present, this is the retired 12-type table; "
                         "cluster 20 is Smooth muscle (see FIG5_NOTEBOOK_HANDOFF.md)")
    return out


_D = load_inputs(os.environ.get("SC_ENDOM_ROOT") or _repo_root_from_here())
PALJ = _D["palj"]
cmap, oe, dsum = _D["cmap"], _D["oe"], _D["dsum"]
sr, cload, edge, lrf = _D["sr"], _D["cload"], _D["edge"], _D["lrf"]

POST = ["E16_5", "PPD1", "PPD7", "PPD30"]
ALL5 = ["CTRL"] + POST
edge_i = {s: edge[edge.stage == s].set_index(["source", "target"]).n for s in ALL5}

render_svg.svg_mode(base=32, small=28, tiny=24)

COMP=dict(zip(cmap.cell_type,cmap.compartment))
COMP_ORDER=['Epithelial','Stromal','Endothelial','Immune']
CPAL={'Epithelial':'#748190','Stromal':'#367f80','Endothelial':'#e58677','Immune':'#c58a4c'}
STAGE_COL={'CTRL':'#8c8c8c','E16_5':'#cb7a29','PPD1':'#A0522D','PPD7':'#FA8072','PPD30':'#4682B4'}
STAGE_LAB={'CTRL':'Control','E16_5':'E16.5','PPD1':'PPD1','PPD7':'PPD7','PPD30':'PPD30'}
MODCOL={'Homeostasis':'#808080','Pregnancy':'#cb7a29','Early repair':'#A0522D',
        'Late repair':'#FA8072','Remodelling':'#4682B4'}
MODOF={'E16_5':'Pregnancy','PPD1':'Early repair','PPD7':'Late repair','PPD30':'Remodelling'}
# Display labels for the pre-final deck. The fifth module is shown as "New Homeostasis";
# its gene set is unchanged (789 genes, peaks PPD30). The published definition tables
# data/raw_light/MODULES_summary.csv and Fig3_module_genes.csv are keyed on "Remodelling",
# so that stays the internal key and the rename happens here, where the label is drawn.
# Do not resolve the inconsistency by rewriting the CSV keys.
DISPLAY={'Remodelling':'New Homeostasis'}


def disp(module):
    """The label a module is drawn with. Identity for the four unrenamed modules."""
    return DISPLAY.get(module, module)


PAL={t:PALJ.get(t,'#999999') for t in COMP}
MX=0.32

TBC={c:[t for t in cmap.cell_type if COMP[t]==c] for c in COMP_ORDER}
GAP=0.15
n=sum(len(v) for v in TBC.values()); span=(2*np.pi-len(COMP_ORDER)*GAP)/n
ANG={}; ARC={}; cur=np.pi/2
for c in COMP_ORDER:
    a0=cur
    for t in TBC[c]: ANG[t]=cur+span/2; cur+=span
    ARC[c]=(a0,cur); cur+=GAP
POS={t:(np.cos(a),np.sin(a)) for t,a in ANG.items()}

M={s:oe[oe.stage==s].pivot(index='sender',columns='receiver',values='log2_oe').reindex(COMP_ORDER)[COMP_ORDER] for s in ALL5}
DELTA={s:(M[s]-M['CTRL']) for s in POST}
srx=sr.copy(); srx['compartment']=srx.cell_type.map(COMP)
srx['x']=srx.stage.map({'CTRL':0,'E16_5':1,'PPD1':2,'PPD7':3,'PPD30':4})
cl=cload.pivot(index='stage',columns='compartment',values='log2_ratio').reindex(POST)[COMP_ORDER]

# Panels h and i. lr_pair packs four fields as ligand|receptor|sender->receiver, so the
# direction split is a string parse, not a separate table.
lrf['ligand']=lrf.lr_pair.str.split('|').str[0]
lrf['absdiff']=lrf.stage_val-lrf.ctrl_val
_pair_ct=lrf.lr_pair.str.split('|').str[2]
lrf['src']=_pair_ct.str.split('->').str[0]
lrf['tgt']=_pair_ct.str.split('->').str[1]

EPI='Epithelial'
NSHOW=140            # bars per row; 140 of ~2500 pairs is one bar per ~2 px at panel width
QCUT=0.35            # drop the weakest third by ctrl+stage total, they carry no share signal


def _share_rows(mask,nshow=NSHOW,qcut=QCUT):
    """Per-stage bar subset and end labels for one direction of epithelial signalling.

    Ligand ranking happens AFTER the total-strength filter, so the labels name genes that
    are actually drawn. nsmallest(6) then dedupe then trim to 5: a ligand can pair with
    several receptors and would otherwise take two label slots.
    """
    rend={}; lab={}
    for s in POST:
        d=lrf[mask&(lrf.stage==s)].copy(); d['tot']=d.ctrl_val+d.stage_val
        d=d[d.tot>=d.tot.quantile(qcut)].sort_values('ctrl_share',ascending=False)
        g=d.groupby('ligand').absdiff.sum()
        lab[s]=(list(dict.fromkeys(g.nsmallest(6).index))[:5],
                list(dict.fromkeys(g.nlargest(6).index))[:5])
        rend[s]=d.iloc[np.linspace(0,len(d)-1,min(nshow,len(d))).astype(int)].reset_index(drop=True)
    return rend,lab


REND_RECV,LAB_RECV=_share_rows(lrf.tgt==EPI)
REND_SEND,LAB_SEND=_share_rows(lrf.src==EPI)

# The recomputed labels must be the ones on the shipped SVGs. If a ranking tie ever flips,
# this fails at import rather than drawing a figure that disagrees with the deck.
_EL=_D['endlab']
for _role,_lab in (('receiver',LAB_RECV),('sender',LAB_SEND)):
    for _s in POST:
        for _i,_side in enumerate(('control','stage')):
            _want=_EL[(_EL.stage==_s)&(_EL.role==_role)&
                      (_EL.side==_side)].ligands.iloc[0].split(', ')
            if _lab[_s][_i]!=_want:
                raise ValueError(f"{_role} {_s} {_side} ligand labels drifted: "
                                 f"{_lab[_s][_i]} vs delivered {_want}")

assert {'DELTA','REND_RECV','REND_SEND','POS','ARC','srx','cl'} <= globals().keys()


RIB_R=1.155
LINE_ALPHA=0.30
LINE_Q=0.75          # draw only the strongest quartile of per-pair changes

def _arcpts(a0,a1,R,n=40):
    th=np.linspace(a0,a1,n); return list(zip(R*np.cos(th),R*np.sin(th)))

def _ribcol(comp,v,gamma=1.15,lo=0.22,hi=1.0):
    base=np.array(to_rgb(CPAL[comp])); white=np.ones(3)
    nn=float(np.clip(v/MX,-1,1))
    return to_hex(white+(base-white)*(lo+(hi-lo)*(((nn+1)/2)**gamma)))

def _ribbon(ax,s0,s1,t0,t1,color,R=RIB_R,zorder=1,pull=0.12):
    ms=(s0+s1)/2; mt=(t0+t1)/2
    c1=(pull*R*np.cos(ms),pull*R*np.sin(ms)); c2=(pull*R*np.cos(mt),pull*R*np.sin(mt))
    v=_arcpts(s0,s1,R); w=_arcpts(t0,t1,R)
    verts=[v[0]]; codes=[MPath.MOVETO]
    for p in v[1:]: verts.append(p); codes.append(MPath.LINETO)
    verts += [c1,c2,w[0]]; codes += [MPath.CURVE4]*3
    for p in w[1:]: verts.append(p); codes.append(MPath.LINETO)
    verts += [c2,c1,v[0]]; codes += [MPath.CURVE4]*3
    verts.append(v[0]); codes.append(MPath.CLOSEPOLY)
    ax.add_patch(mpl.patches.PathPatch(MPath(verts,codes),facecolor=color,edgecolor='white',
                                       lw=0.55,alpha=1.0,zorder=zorder))

def _slots(stage,pad=0.045,topn=8):
    Dl=DELTA[stage]
    pairs=sorted([(a,b) for a in COMP_ORDER for b in COMP_ORDER],
                 key=lambda k:-abs(Dl.loc[k[0],k[1]]))[:topn]
    W={c:[] for c in COMP_ORDER}
    for a,b in pairs:
        w=abs(Dl.loc[a,b]); W[a].append(('out',b,w))
        W[b if a!=b else a].append(('in',a,w))
    out={}; inc={}
    for c in COMP_ORDER:
        a0,a1=ARC[c]; a0+=pad; a1-=pad
        tot=sum(w for _,_,w in W[c]) or 1.0; cur=a0
        for kind,other,w in W[c]:
            sp=(a1-a0)*w/tot
            (out if kind=='out' else inc)[((c,other) if kind=='out' else (other,c))]=(cur,cur+sp)
            cur+=sp
    return out,inc,pairs

def arclab(ax,c,a0,a1,R=1.50,fs=7.0):
    am=(a0+a1)/2; sp=a1-a0
    kw=dict(color=CPAL[c],fontsize=fs,ha='center',va='center',zorder=7,fontweight='bold')
    if sp<0.55: ax.text(R*np.cos(am),R*np.sin(am),c,**kw); return
    deg=np.degrees(am)%360; rot=(deg-90)%360
    if 90<rot<270: rot=(rot+180)%360
    ax.text(R*np.cos(am),R*np.sin(am),c,rotation=rot,**kw)

def circle16(ax,stage,fs=7.0,node_s=70,arc_lw=4.0,node_ec=0.9,line_max=2.4):
    Dl=DELTA[stage]; out,inc,pairs=_slots(stage)
    for a,b in sorted(pairs,key=lambda k:abs(Dl.loc[k[0],k[1]])):
        s0,s1=out[(a,b)]; t0,t1=inc[(a,b)]
        _ribbon(ax,s0,s1,t0,t1,_ribcol(a,Dl.loc[a,b]),zorder=2)
    A=edge_i['CTRL']; B=edge_i[stage]
    dvc=[(k,int(B.get(k,0))-int(A.get(k,0))) for k in sorted(set(A.index)|set(B.index))
         if k[0]!=k[1] and k[0] in POS and k[1] in POS]
    mag=np.array([abs(v) for _,v in dvc]); thr=np.percentile(mag[mag>=1],LINE_Q*100)
    mxc=mag.max() or 1
    for (s,t),v in dvc:
        if abs(v)<thr: continue
        x1,y1=POS[s]; x2,y2=POS[t]
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),zorder=3,
            arrowprops=dict(arrowstyle='-',color='#8a8a8a',lw=0.25+line_max*abs(v)/mxc,
                            alpha=LINE_ALPHA,connectionstyle='arc3,rad=0.16'))
    for c in COMP_ORDER:
        a0,a1=ARC[c]; th=np.linspace(a0,a1,90)
        ax.plot(1.30*np.cos(th),1.30*np.sin(th),color=CPAL[c],lw=arc_lw,solid_capstyle='butt',zorder=5)
        arclab(ax,c,a0,a1,fs=fs)
    for t,(x,y) in POS.items():
        ax.scatter([x],[y],s=node_s,color=PAL[t],edgecolors='white',linewidths=node_ec,zorder=6)
    ax.set_xlim(-1.72,1.72); ax.set_ylim(-1.72,1.66); ax.set_aspect('equal'); ax.axis('off')



def leaders(ax,rows,fs=5.2):
    pr=ax.get_proj(); xy=[]
    for _,r in rows.iterrows():
        x2,y2,_=proj3d.proj_transform(r.x,r.sender_mean_lr,r.receiver_mean_lr,pr)
        pt=ax.transAxes.inverted().transform(ax.transData.transform((x2,y2)))
        xy.append((pt[0],pt[1],r))
    xy.sort(key=lambda z:z[0])
    for (xf,yf),(px,py,r) in zip([(-0.04,1.00),(0.44,1.14)],xy):
        ax.annotate(f"{STAGE_LAB[r.stage]} {r.cell_type}\n({int(r.sender_count)})",
            xy=(px,py),xycoords='axes fraction',xytext=(xf,yf),textcoords='axes fraction',
            color=PAL[r.cell_type],fontsize=fs,ha='left',va='bottom',
            arrowprops=dict(arrowstyle='-',color=PAL[r.cell_type],lw=0.7,shrinkA=1,shrinkB=2,alpha=0.85),zorder=8)

def panel3d(ax,c,fs=6.0,ms=16,stem_lw=0.5,mew=0.35):
    d=srx[srx.compartment==c].dropna(subset=['sender_mean_lr','receiver_mean_lr'])
    for t in sorted(d.cell_type.unique()):
        dt=d[d.cell_type==t]
        for _,r in dt.iterrows():
            ax.plot([r.x,r.x],[r.sender_mean_lr,r.sender_mean_lr],[0,r.receiver_mean_lr],
                    color=PAL[t],lw=stem_lw,alpha=0.8,zorder=1,solid_capstyle='round')
        ax.scatter(dt.x,dt.sender_mean_lr,dt.receiver_mean_lr,s=ms,color=PAL[t],
                   edgecolors='white',linewidths=mew,depthshade=False,zorder=3)
    ax.set_xlabel('Timepoint',labelpad=1,fontsize=fs); ax.set_ylabel('Sender',labelpad=1,fontsize=fs)
    ax.set_zlabel('Receiver',labelpad=1,fontsize=fs)
    ax.set_xticks([0,1,2,3,4]); ax.set_xticklabels(['C','E','1','7','30'])
    ax.yaxis.set_major_locator(MaxNLocator(3)); ax.zaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=fs-1.5,pad=0)
    ax.set_zlim(0,d.receiver_mean_lr.max()*1.30); ax.view_init(elev=18,azim=-62)
    ax.set_box_aspect((1.0,1.0,0.80))
    for a in (ax.xaxis,ax.yaxis,ax.zaxis):
        a.pane.set_alpha(0.04); a._axinfo['grid'].update(color='#dddddd',linewidth=0.5)
    ax.text2D(0.5,-0.02,c,transform=ax.transAxes,fontsize=fs+1.5,color='white',ha='center',va='top',
              bbox=dict(boxstyle='round,pad=0.34',fc=CPAL[c],ec='none'))
    leaders(ax,d.nlargest(2,'sender_count'),fs=fs-1.0)

def traj(ax,fs=8):
    for c in COMP_ORDER:
        y=[0.0]+[cl.loc[s,c] for s in POST]
        ax.plot(range(5),y,'-o',color=CPAL[c],ms=fs*0.62,lw=fs*0.22,label=c,mec='white',mew=1.1,zorder=3)
    ax.axhline(0,color='#c0392b',lw=1.2,ls='--',zorder=1)
    for i,s in enumerate(POST,start=1):
        ax.axvspan(i-0.32,i+0.32,color=STAGE_COL[s],alpha=0.10,lw=0,zorder=0)
    ax.set_xticks(range(5)); ax.set_xticklabels([STAGE_LAB[s] for s in ALL5],fontsize=fs-1)
    ax.set_ylabel('Signalling load\n(log$_2$ vs control)',fontsize=fs)
    ax.tick_params(labelsize=fs-1); ax.set_xlim(-0.35,4.45); ax.set_ylim(-0.30,1.62)
    ax.text(-0.30,0.03,'control',fontsize=fs-3,color='#c0392b',va='bottom',ha='left')
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False,fontsize=fs-3,loc='upper center',bbox_to_anchor=(0.5,1.02),
              ncol=4,handlelength=1.4,columnspacing=1.2,borderpad=0.05)

def oepanel(ax,fs=8):
    S=dsum.set_index('stage').reindex(ALL5)
    for col,lab,c in [('epi_to_epi','Epithelial→Epi',CPAL['Epithelial']),
                      ('stroma_to_epi','Stromal→Epi',CPAL['Stromal']),
                      ('endo_to_epi','Endothelial→Epi',CPAL['Endothelial']),
                      ('immune_to_epi','Immune→Epi','#8c3b2e')]:
        ax.plot(range(5),S[col],'-o',color=c,ms=fs*0.62,lw=fs*0.22,label=lab,mec='white',mew=1.1)
    ax.axhline(0,color='#888',lw=1.0,ls='--')
    ax.set_xticks(range(5)); ax.set_xticklabels([STAGE_LAB[s] for s in ALL5],fontsize=fs-1)
    ax.set_ylabel('Preference toward\nepithelium (log$_2$ O/E)',fontsize=fs)
    ax.tick_params(labelsize=fs-1); ax.set_xlim(-0.35,4.5); ax.set_ylim(-0.46,0.32)
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False,fontsize=fs-4.5,loc='lower center',bbox_to_anchor=(0.5,-0.02),
              ncol=2,handlelength=1.4,columnspacing=1.2,borderpad=0.05,labelspacing=0.25)

def share_panel(ax,stage,rend,lab,fs=8.0):
    d=rend[stage]; x=np.arange(len(d))
    ax.bar(x,d.ctrl_share,width=1.0,color='#9e9e9e',lw=0)
    ax.bar(x,1-d.ctrl_share,bottom=d.ctrl_share,width=1.0,color=STAGE_COL[stage],lw=0)
    ax.axhline(0.5,color='white',lw=0.7,zorder=4)
    ax.set_xlim(-0.5,len(d)-0.5); ax.set_ylim(0,1); ax.set_xticks([])
    ax.set_yticks([0,0.5,1]); ax.tick_params(labelsize=fs-1.5)
    ax.set_ylabel('Share of\nsignal',fontsize=fs)
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    tail=d[d.ctrl_share<0.5]
    if len(tail):
        ax.add_patch(mpl.patches.Rectangle((tail.index.min()-0.5,0.012),len(tail),0.976,
                     fill=False,ec=STAGE_COL[stage],lw=0.9,ls=(0,(3,2)),zorder=5))
    ax.text(-0.5,-0.055,', '.join(lab[stage][0]),fontsize=fs-5,color='#777',ha='left',va='top',style='italic')
    ax.text(len(d)-0.5,-0.055,', '.join(lab[stage][1]),fontsize=fs-5,color=STAGE_COL[stage],
            ha='right',va='top',style='italic')
    ax.text(0.0,1.03,'Control',transform=ax.transAxes,color='#666',fontsize=fs,fontweight='bold',ha='left',va='bottom')
    ax.text(0.075,1.03,'vs',transform=ax.transAxes,color='#999',fontsize=fs-2,ha='left',va='bottom')
    ax.text(0.115,1.03,STAGE_LAB[stage],transform=ax.transAxes,color=STAGE_COL[stage],
            fontsize=fs+1,fontweight='bold',ha='left',va='bottom')
    ax.text(1.0,1.03,disp(MODOF[stage]),transform=ax.transAxes,fontsize=fs-2,
            color=MODCOL[MODOF[stage]],ha='right',va='bottom',style='italic')

def letter(ax,ch,dx=-0.06,dy=1.02,fs=11):
    ax.text(dx,dy,ch,transform=ax.transAxes,fontweight='bold',fontsize=fs,va='bottom',ha='left')

def rib_key(fig,rect,fs=12):
    ax=fig.add_axes(rect); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    for i,c in enumerate(COMP_ORDER):
        y=0.78-i*0.225
        for j in range(6):
            v=-MX+2*MX*j/5
            ax.add_patch(mpl.patches.Rectangle((0.145+j*0.034,y-0.072),0.034,0.144,
                         fc=_ribcol(c,v),ec='white',lw=0.4))
        ax.text(0.135,y,c,fontsize=fs-1,color=CPAL[c],ha='right',va='center',fontweight='bold')
    ax.annotate('',xy=(0.349,0.965),xytext=(0.145,0.965),arrowprops=dict(arrowstyle='->',color='#888',lw=1.0))
    ax.text(0.247,1.02,'decreased  →  increased vs control',fontsize=fs-2.5,color='#666',ha='center',va='bottom')
    ax.text(0.247,-0.14,'ribbon = compartment-level directional preference',fontsize=fs-2,color='#555',ha='center',va='top')
    ax.plot([0.46,0.55],[0.66,0.66],color='#8a8a8a',lw=0.5,alpha=LINE_ALPHA)
    ax.plot([0.46,0.55],[0.40,0.40],color='#8a8a8a',lw=2.6,alpha=LINE_ALPHA)
    ax.text(0.567,0.53,'grey lines = strongest quartile of cell-type pairs,\nthickness = |change in interactions vs control|',
            fontsize=fs-2,color='#555',va='center')

def share_block(f,gsub,letter_ch,rend,lab,title,fs=20):
    """One share-of-signal panel: four control-vs-stage rows under a single letter."""
    g=gsub.subgridspec(4,1,hspace=0.72)
    axes=[]
    for k,s in enumerate(POST):
        ax=f.add_subplot(g[k,0]); share_panel(ax,s,rend,lab,fs=fs); axes.append(ax)
    letter(axes[0],letter_ch,dx=-0.078,dy=1.42,fs=26)
    axes[0].text(0.5,1.42,title,transform=axes[0].transAxes,fontsize=fs+1,
                 fontweight='bold',ha='center',va='bottom')
    return axes


def build6():
    f=plt.figure(figsize=(19.5,32.4))
    gs=f.add_gridspec(5,1,height_ratios=[2.05,1.62,1.24,2.55,2.55],hspace=0.46,
                      left=0.092,right=0.980,top=0.918,bottom=0.024)
    g0=gs[0].subgridspec(1,4,wspace=0.10)
    for i,c in enumerate(COMP_ORDER):
        ax=f.add_subplot(g0[0,i],projection='3d'); panel3d(ax,c,fs=16,ms=110,stem_lw=2.8,mew=1.0)
        ax.text2D(-0.04,1.26,'abcd'[i],transform=ax.transAxes,fontweight='bold',fontsize=26,va='bottom')
    g1=gs[1].subgridspec(1,4,wspace=0.02)
    circ=[]
    for j,s in enumerate(POST):
        ax=f.add_subplot(g1[0,j]); circle16(ax,s,fs=14.5,node_s=165,arc_lw=7.0,node_ec=1.3,line_max=2.4)
        ax.set_title(f"Control vs {STAGE_LAB[s]}",fontsize=21,color=STAGE_COL[s],fontweight='bold',pad=3)
        circ.append(ax)
    # The four discs are one panel in the deck, so the letter sits once on the leftmost.
    letter(circ[0],'e',dx=0.01,dy=1.10,fs=26)
    g2=gs[2].subgridspec(1,2,wspace=0.28)
    axf=f.add_subplot(g2[0,0]); traj(axf,fs=20); letter(axf,'f',dx=-0.130,dy=1.06,fs=26)
    axg=f.add_subplot(g2[0,1]); oepanel(axg,fs=20); letter(axg,'g',dx=-0.130,dy=1.06,fs=26)
    share_block(f,gs[3],'h',REND_RECV,LAB_RECV,'Epithelium as receiver  (all cell types → Epithelial)')
    share_block(f,gs[4],'i',REND_SEND,LAB_SEND,'Epithelium as sender  (Epithelial → all cell types)')
    h=[plt.Line2D([],[],marker='o',ls='',ms=10,color=PAL[t],label=t) for t in cmap.cell_type]
    f.legend(handles=h,loc='upper center',ncol=8,fontsize=15,frameon=False,
             bbox_to_anchor=(0.5,0.984),handletextpad=0.35,columnspacing=1.4)
    p=circ[0].get_position()
    rib_key(f,[0.095,p.y0-0.040,0.44,0.034],fs=15)
    # Was 'diverges from control in four stage-specific waves', which asserts a
    # monotonic drift the data does not show: Jaccard similarity to control runs
    # 0.290 / 0.356 / 0.431 / 0.347, rising through PPD7 before falling back. The
    # similarity measure now lives in SFig 4e, not on this sheet, so the title
    # rests on panel f instead: log2 load vs control at PPD30 is Epithelial 0.956,
    # Stromal 0.933, Endothelial 0.413, Immune 1.136, all four above control.
    # 'Elevated' not 'peaks': endothelium's own maximum is E16.5 (0.525), not PPD30.
    f.suptitle('Inferred signalling load is elevated in every compartment during New Homeostasis (PPD30)',
               fontsize=28,y=0.997)
    return f



def build(repo_root, out_dir):
    """Render Fig 6 as one composite SVG; returns {panel: path} for panels a-i."""
    fig = build6()
    render_svg.transparent(fig)
    stem = os.path.join(out_dir, "extra_signalling_vs_control_16types")
    render_svg.save_svg(fig, stem, also_pdf=True)
    v = render_svg.verify(stem + ".svg")
    if not v["text_elements"]:
        raise RuntimeError(f"Fig6 exported with no live text: {v}")
    plt.close(fig)
    return {ch: stem + ".svg" for ch in "abcdefghi"}

"""Pure diagnostics for unchanged-model numerical quantile refinement.

No I/O, observation draws, inference, tuning or interval adjustment. Candidate
polynomials and supplied targets are explicit; oracle errors describe table and
floating representation error, not a new physical solve or population coverage.
"""
from fractions import Fraction
import numpy as np
from experiments.campaign_c38_table_validation import POINTS
from experiments.campaign_c38_controlled_observations import POINT_IDS
PROBABILITIES=(.025,.5,.975)
CDF_THRESHOLDS=(1e-6,1e-4,.01)
METHODS=('adaptive','GL128')


def polynomial_ranges(coefficients):
 """Exact binary-rational extrema over x=75000/E in[1/2,1], all categories."""
 arrays={k:np.asarray(v,dtype=float)for k,v in coefficients.items()}
 if not arrays or any(v.shape!=(3,189)or not np.isfinite(v).all()for v in arrays.values()):raise ValueError('Finite full63 canonical polynomial coefficients')
 ranges=[];nonconstant=[]
 for j in range(189):
  values=[]
  for a in arrays.values():
   c=[Fraction(float(v))for v in a[:,j]];x=[Fraction(1,2),Fraction(1)]
   if c[2]:
    critical=-c[1]/(2*c[2])
    if x[0]<critical<x[1]:x.append(critical)
   values.extend(c[0]+t*(c[1]+t*c[2])for t in x)
  lo,hi=min(values),max(values);ranges.append(float(hi-lo));nonconstant.append(lo!=hi)
 return dict(range_m=np.array(ranges).reshape(63,3),nonconstant=np.array(nonconstant).reshape(63,3),scope='Exact rational extrema of stored binary64 target polynomials; not a bound on physical/FEM forward error.')


def quantile_diagnostics(prediction,topology):
 mask=np.asarray(topology['nonconstant'])[POINT_IDS];result={}
 for name in METHODS:
  row=prediction.get('methods',{}).get(name,{})
  entries=row.get('canonical_quantile_details',[]);seen=set();qrows=[];kinds={}
  for entry in entries:
   point,axis=entry['point_id'],entry['axis'];key=(point,axis)
   if key in seen or point not in POINT_IDS or axis not in (0,1,2):raise ValueError('Unique qualified9x3 quantile coordinates')
   seen.add(key);qs=entry.get('quantiles',[])
   if len(qs)>3:raise ValueError('Exactly2.5/50/97.5 quantile slots')
   kinds[key]=[q.get('kind')for q in qs]
   for probability,q in zip(PROBABILITIES,qs):
    kind=q.get('kind','');atom=kind in ('common_constant_atom','category_mixture_atom');cdf=q.get('CDF_at',{});left=q.get('CDF_left',{});values=[cdf.get(k,float('nan'))for k in ('lower','cdf','upper')];ordered=bool(np.isfinite(values).all()and 0<=values[0]<=values[1]<=values[2]<=1)
    bound=max(abs(values[0]-probability),abs(values[2]-probability))if np.isfinite(values).all()else None
    atomqualified=bool(atom and np.isfinite([left.get('upper',float('nan')),cdf.get('lower',float('nan'))]).all()and left['upper']<probability<=cdf['lower'])
    qrows.append(dict(point_id=int(point),axis=int(axis),probability=probability,kind=kind,atom=atom,CDF_enclosure_ordered=ordered,maximum_CDF_error_bound=None if atom else bound,atom_jump_qualified=atomqualified if atom else None))
  continuous=[r for r in qrows if not r['atom']];bounds=[r['maximum_CDF_error_bound']for r in continuous if r['maximum_CDF_error_bound']is not None]
  r=dict(method_completed=row.get('completed')is True,expected_coordinates=27,available_coordinate_entries=len(seen),expected_quantiles=81,available_quantiles=len(qrows),missing_quantiles=81-len(qrows),continuous_quantiles=len(continuous),atom_quantiles=len(qrows)-len(continuous),invalid_continuous_CDF_enclosures=sum(not x['CDF_enclosure_ordered']for x in continuous),missing_continuous_CDF_bounds=sum(x['maximum_CDF_error_bound']is None for x in continuous),maximum_continuous_CDF_error_bound=max(bounds)if bounds else None,continuous_CDF_error_count_greater_than={str(t):sum(x>t for x in bounds)for t in CDF_THRESHOLDS},unqualified_atom_jumps=sum(x['atom']and not x['atom_jump_qualified']for x in qrows),quantiles=qrows)
  for label,key in [('displacement','canonical_displacement_equal_tail_m'),('position','canonical_position_equal_tail_m')]:
   if key not in row:r[label]=dict(available_coordinates=0,missing_coordinates=27);continue
   intervals=np.asarray(row[key],dtype=float)
   if intervals.shape!=(9,3,3)or not np.isfinite(intervals).all():raise ValueError('Finite9x3x3 saved intervals')
   width=intervals[:,:,2]-intervals[:,:,0];zero=width==0;atom_bounds=np.zeros((9,3),bool);continuous_bounds=np.zeros((9,3),bool)
   for a,point in enumerate(POINT_IDS):
    for axis in range(3):
     k=kinds.get((point,axis),[])
     if len(k)==3:atom_bounds[a,axis]=k[0]in('common_constant_atom','category_mixture_atom')and k[2]in('common_constant_atom','category_mixture_atom');continuous_bounds[a,axis]=not atom_bounds[a,axis]
   r[label]=dict(available_coordinates=27,missing_coordinates=0,negative_width_count=int((width<0).sum()),zero_width_count=int(zero.sum()),zero_width_nonconstant_count=int((zero&mask).sum()),zero_width_nonconstant_continuous_bounds_count=int((zero&mask&continuous_bounds).sum()),zero_width_nonconstant_atom_bounds_count=int((zero&mask&atom_bounds).sum()),zero_width_constant_count=int((zero&~mask).sum()),maximum_width_m=float(width.max()),minimum_width_m=float(width.min()))
  result[name]=r
 return result


def core_group_status(proof):
 paths=proof['bit_mismatch_paths']
 groups=dict(material={'material_CDF','material_quantile_details','material_equal_tail_Pa'},mean={'canonical_mean_displacement_m','canonical_mean_positions_m'},variance={'canonical_coordinate_variance_m2','canonical_mean_coordinate_variance_m2','canonical_moment_numerics','canonical_variance_integration_resolution_estimate_m2','canonical_variance_resolution_scope'})
 return dict(all_core_bit_identical=proof['passed'],**{name:not any(any(key in p.split('/')for key in keys)for p in paths)for name,keys in groups.items()})


def oracle_error(coefficients,young_Pa,actual):
 """Compare table interpolation at supplied truephysics, not refit or newFEM."""
 c=np.asarray(coefficients,dtype=float);q=np.asarray(actual,dtype=float)
 if c.shape!=(3,189)or q.shape!=(63,3)or not np.isfinite(c).all()or not np.isfinite(q).all()or not 75000<=young_Pa<=150000:raise ValueError('Finitecanonicalq63 and declared material range')
 x=75000/float(young_Pa);u=(c[0]+x*(c[1]+x*c[2])).reshape(63,3);target=q-POINTS;du=u-target;dq=(POINTS+u)-q
 def stats(a):return dict(coordinate_RMS_by_axis_m=np.sqrt(np.mean(a*a,axis=0)),maximum_absolute_by_axis_m=np.max(abs(a),axis=0),coordinate_RMS_m=float(np.sqrt(np.mean(a*a))),maximum_point_norm_m=float(np.linalg.norm(a,axis=1).max()))
 return dict(displacement_error_m=du,position_error_m=dq,all63_displacement=stats(du),selected9_displacement=stats(du[POINT_IDS]),all63_position=stats(dq),selected9_position=stats(dq[POINT_IDS]),position_vs_displacement_error_roundtrip_max_m=float(abs(dq-du).max()),scope='Public candidate canonical polynomial at supplied true E/BC versus direct withheld FEM target. Includes interpolation, discretization/source representation and binary64 subtraction/addition; no new solve or rigorous continuous-error certificate.')


def error_relative_halfwidth(prediction,oracle):
 row=prediction.get('methods',{}).get('adaptive',{});key='canonical_displacement_equal_tail_m'
 if key not in row:return dict(available_coordinates=0,missing_coordinates=27,scope='No available saved adaptive displacement interval')
 q=np.asarray(row[key]);error=abs(np.asarray(oracle['displacement_error_m'])[POINT_IDS]);half=(q[:,:,2]-q[:,:,0])/2
 if half.shape!=(9,3)or not np.isfinite(half).all():raise ValueError('Finite saved displacement halfwidth')
 positive=half>0;ratio=error[positive]/half[positive];zero=half==0
 return dict(available_coordinates=27,missing_coordinates=0,positive_halfwidth_count=int(positive.sum()),negative_halfwidth_count=int((half<0).sum()),zero_halfwidth_zero_error_count=int((zero&(error==0)).sum()),zero_halfwidth_positive_oracle_error_count=int((zero&(error>0)).sum()),finite_ratio_count=len(ratio),maximum_finite_error_over_halfwidth=float(ratio.max())if len(ratio)else None,median_finite_error_over_halfwidth=float(np.median(ratio))if len(ratio)else None,positive_halfwidth_error_larger_than_halfwidth_count=int((error[positive]>half[positive]).sum()),scope='Descriptive public-model oracle error divided by raw adaptive displacement95% interval halfwidth at9points; no noise addition, correction, widening or coverage guarantee.')

"""Prospective scalar roundoff resolution of the frozen nested-contact test.

Constructed public rest geometry only. No saved fields/contact integration.
These conservative forward bounds diagnose resolution; they do not alter gates.
"""
from fractions import Fraction as Q
import math
import numpy as np
from experiments import campaign_contact_nested_mesh as nested


def exact_axis_aligned_area(vertices):
 X=[[Q(float(v))for v in row]for row in vertices];a=[X[1][j]-X[0][j]for j in range(3)];b=[X[2][j]-X[0][j]for j in range(3)];cross=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
 if sum(v!=0 for v in cross)!=1:raise ValueError('Only proved axis-aligned nondegenerate box faces')
 return sum(abs(v)for v in cross)/2


def gamma(operations):
 eps=np.finfo(float).eps
 if not 0<=operations*eps<1:raise ValueError('Finite FP operation budget required')
 return operations*eps/(1-operations*eps)


def height_gradient_bound(delta_height,area=.0896,kappa=1e9):
 # Coarse T6 L1 Lebesgue constant <=5/3 implies its Euclidean norm bound.
 return kappa*area*(5/3)*delta_height


def interpret(raw_difference,roundoff_bound,gate=1e-10):
 if raw_difference<=gate:return 'PASS_ORIGINAL_RAW_GATE'
 if raw_difference<=roundoff_bound:return 'FAIL_ORIGINAL_RAW_GATE_CAUSE_UNRESOLVED_BY_INPUT_REDUCTION_RESOLUTION'
 return 'FAIL_ORIGINAL_RAW_GATE_EXCEEDS_STATED_INPUT_REDUCTION_BOUND_OTHER_ERRORS_STILL_REQUIRE_AUDIT'


def scalar_resolution():
 coarse,fine=nested.geometry((4,2,2)),nested.geometry((8,4,4));P,proof=nested.nested_transfer(coarse,fine);C=nested.lattice(coarse)*2;F=nested.lattice(fine)
 cf,ca=nested.surface_geometry(coarse);ff,fa=nested.surface_geometry(fine);exactC=[exact_axis_aligned_area(coarse.rest[f[:3]])for f in cf];exactF=[exact_axis_aligned_area(fine.rest[f[:3]])for f in ff];parents=[]
 for face in fine.faces:
  V=F[face];matches=[]
  for k,coarse_face in enumerate(coarse.faces):
   tri=C[coarse_face];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0])
   if np.any((V-tri[0])@normal):continue
   D=np.column_stack([tri[1]-tri[0],tri[2]-tri[0]]);uv=np.linalg.lstsq(D,(V-tri[0]).T,rcond=None)[0].T;uv=np.rint(uv*4)/4
   if np.array_equal(uv@D.T+tri[0],V)and np.all(uv>=0)and np.all(uv.sum(1)<=1):matches.append(k)
  if len(matches)!=1:raise ValueError('Unique certified coarse face required')
  parents.append(matches[0])
 if any(parents.count(k)!=4 for k in range(len(cf))):raise ValueError('Four equal-material-area children per coarse triangle required')
 area_model_variation=sum(abs(A-exactC[k]/4)for A,k in zip(exactF,parents))
 area_evaluation_error=sum(abs(Q(float(a))-A)for a,A in zip(ca,exactC))+sum(abs(Q(float(a))-A)for a,A in zip(fa,exactF))
 geometry_error=Q(0);rest_dot_error=Q(0);rounded=P@coarse.rest
 for row in range(P.shape[0]):
  sl=slice(P.indptr[row],P.indptr[row+1]);indices=P.indices[sl];values=P.data[sl]
  for j in range(3):
   exact=sum(Q(float(w))*Q(float(coarse.rest[k,j]))for w,k in zip(values,indices));geometry_error=max(geometry_error,abs(exact-Q(float(fine.rest[row,j]))));rest_dot_error=max(rest_dot_error,abs(Q(float(rounded[row,j]))-exact))
 row_terms=int(np.diff(P.indptr).max());column_terms=int(np.diff(P.tocsc().indptr).max());Pinf=float(np.max(abs(P).sum(1)));Pone=float(np.max(abs(P).sum(0)));A=float(sum(exactC));geometry=float(geometry_error)
 dot_per_H=gamma(2*row_terms)*Pinf;area_delta=float(area_model_variation+area_evaluation_error);surface_height_per_H=(5/3)*Pinf;area_per_H=1e9*(5/3)*surface_height_per_H*area_delta
 scenarios=[]
 for H in (.1,.2,1.):
  delta_field=(5/3)*(dot_per_H*H+geometry);surface_height=surface_height_per_H*H
  input_bound=height_gradient_bound(delta_field,area=A)+area_per_H*H
  energy_bound=1e9*A*(surface_height*delta_field+.5*delta_field**2)+.5e9*surface_height**2*area_delta
  scenarios.append(dict(assumed_maximum_coarse_nodal_height_m=H,height_and_area_gradient_bound_N=input_bound,ratio_to_fixed_gradient_gate=input_bound/1e-10,height_and_area_energy_bound_J=energy_bound,ratio_to_fixed_energy_gate=energy_bound/1e-12,not_a_bound_for_saved_fields_until_height_ceiling_verified=True))
 return dict(constructed_geometry_only=True,saved_fields_read=False,contact_integrations=0,original_gradient_gate_N=1e-10,original_energy_gate_J=1e-12,actual_P_row_terms_max=row_terms,actual_P_column_terms_max=column_terms,P_abs_infinity_norm=Pinf,P_abs_one_norm=Pone,total_reference_area_m2=A,exact_material_nesting=proof['exact_dyadic_material_partition'],exact_physical_geometry_mismatch_Linf_m=str(geometry_error),physical_geometry_mismatch_Linf_m=geometry,measured_rest_prolongation_dot_roundoff_Linf_m=float(rest_dot_error),rest_roundoff_is_only_a_proxy_for_actual_saved_field=True,generic_prolongation_height_error_m_per_height_m=dot_per_H,reference_area_model_total_variation_m2=str(area_model_variation),reference_area_evaluation_error_L1_m2=str(area_evaluation_error),gradient_bound_from_geometry_mismatch_N=height_gradient_bound((5/3)*geometry,area=A),gradient_bound_from_area_per_height_N_m=area_per_H,restriction_reduction_error_bound_per_fine_abs_gradient_L2=gamma(2*column_terms)*math.sqrt(Pinf*Pone),height_scenarios=scenarios,scope='Conservative operation-count forward estimates for FP input/prolongation/area/restriction only. Adaptive/fixed-rule numerical integration errors remain separate; no interval certificate for all arithmetic.',decision='Raw1e-10N/1e-12J gates unchanged. A raw failure inside a conservative input/reduction bound is unresolved causal evidence, not an operator-bug proof and not a waived PASS. No residual subtraction or threshold widening.')

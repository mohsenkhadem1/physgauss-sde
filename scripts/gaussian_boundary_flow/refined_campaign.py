"""Reuse the tested training/evaluation contract with finer forward mechanics."""
import sys
from scripts.gaussian_boundary_flow import campaign
from scripts.gaussian_boundary_flow.refined import RefinedBlock
campaign.Block=RefinedBlock
if __name__=='__main__':
    if '--out' not in sys.argv:sys.argv+=['--out','runs/gaussian-boundary-flow-refined-v02']
    campaign.main()

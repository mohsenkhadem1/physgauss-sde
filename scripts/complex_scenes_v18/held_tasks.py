"""Explicit dependency graph for the complete held campaign; no job starts here."""
from pathlib import Path
from scripts.complex_scenes_v18.run_development import ROOT


def build(protocol,public,campaign,freeze_path):
    campaign=Path(campaign).resolve();freeze_path=Path(freeze_path).resolve()
    relative=campaign.relative_to((ROOT/'runs/complex-scenes-v18').resolve())
    main=str(ROOT/'.venv/bin/python');native=str(ROOT/'runs/complex-scenes-v18/baseline-env/bin/python');tasks=[]
    def add(key,kind,command,deps=(),**metadata):
        tasks.append(dict(id=key,kind=kind,argv=command,dependencies=list(deps),**metadata))
    for world in public['worlds']:
        name=world['world_id'];family=world['family'];prefix=campaign/'prefixes'/name
        refbase=[main,'-m','scripts.complex_scenes_v18.held_reference','--campaign',str(campaign),'--freeze',str(freeze_path),'--world-id',name]
        calibration=f'{name}:reference:calibration'
        add(calibration,'cpu',refbase+['--slot','calibration'],world_id=name,role='reference')
        n=protocol['reference']['mesh_n'];dt=protocol['reference']['dt_s'];surface=0 if family=='deformable_pair'else 5;variant=world['calibration']['variant']
        reference=campaign/'references'/name/'calibration'/f'{family}-v{variant}-n{n}-dt{dt:.8f}-s{surface}'
        export=f'{name}:prefix'
        add(export,'cpu',[main,'-m','scripts.complex_scenes_v18.export_public_prefix','--reference',str(reference),'--output',str(prefix),'--cut',str(world['calibration_duration_s']),'--split','held'],[calibration],world_id=name,role='prefix')
        for index in range(len(world['queries'])):
            add(f'{name}:reference:{index}','cpu',refbase+['--slot',str(index)],world_id=name,role='reference')
            if world['refinement']:add(f'{name}:reference:{index}:refined','cpu',refbase+['--slot',str(index),'--refine'],world_id=name,role='reference-refinement')
        for method,settings in protocol['methods'].items():
            if family not in settings['families']:continue
            label=relative/'fits'/method/name
            if method in ('ours','analytic-control'):
                stages=settings['evaluation_stages'];previous=None;dep=export
                if not stages or any(not isinstance(v,int)or v<1 for v in stages):raise ValueError('Explicit positive material fit stages required')
                for stage,evaluations in enumerate(stages):
                    stage_label=label/f'stage{stage}';key=f'{name}:{method}:fit:{stage}'
                    command=[main,'-m','scripts.complex_scenes_v18.material_scene_fit','--case',family,'--variant',str(variant),'--prefix',str(prefix),'--label',str(stage_label),'--evaluations',str(evaluations),'--law-type','learned'if method=='ours'else'analytic']
                    if previous:command+=['--initialize-fit',str(previous)]
                    add(key,'cpu',command,[dep],world_id=name,method=method,role='fit');dep=key
                    previous=ROOT/'runs/complex-scenes-v18'/stage_label/f'{family}-v{variant}'
                fitdir=previous;fitkey=dep
            else:
                fitkey=f'{name}:{method}:fit';fitdir=ROOT/'runs/complex-scenes-v18'/label/f'{family}-v{variant}'
                if method=='uniphy':
                    command=[native,'-m','scripts.complex_scenes_v18.uniphy_latent_fit','--variant',str(variant),'--prefix',str(prefix),'--label',str(label),'--epochs',str(settings['epochs']),'--lr',str(settings['learning_rate']),'--seconds',str(settings['fit_seconds'])]
                    if settings['screen_centroids']:command+=['--screen-centroids']
                    if settings['corrected_adjoint']:command+=['--corrected-adjoint']
                else:
                    command=[native,'-m','scripts.complex_scenes_v18.spring_edge_fit','--method',method,'--case',family,'--variant',str(variant),'--prefix',str(prefix),'--label',str(label),'--epochs',str(settings['epochs']),'--lr',str(settings['learning_rate']),'--seconds',str(settings['fit_seconds'])]
                add(fitkey,'gpu',command,[export],world_id=name,method=method,role='fit')
            for index in range(len(world['queries'])):
                command=[main if method in ('ours','analytic-control')else native,'-m','scripts.complex_scenes_v18.held_prediction','--campaign',str(campaign),'--fitdir',str(fitdir),'--world-id',name,'--index',str(index),'--method',method]
                add(f'{name}:{method}:predict:{index}','cpu'if method in ('ours','analytic-control')else'gpu',command,[fitkey],world_id=name,method=method,role='prediction',query_index=index)
    ids=[t['id']for t in tasks]
    if len(ids)!=len(set(ids))or any(dep not in ids for t in tasks for dep in t['dependencies']):raise ValueError('Invalid task graph identity')
    return tasks

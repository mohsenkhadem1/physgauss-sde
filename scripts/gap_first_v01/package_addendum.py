"""Package a self-contained addendum source without changing earlier releases."""
from pathlib import Path
import hashlib,json,re,zipfile,sys

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'papers/cvpr2027-joint-identification'


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from campaign_report import validate_completion
    completion=validate_completion()
    required=['gap-first-addendum.tex','sections/gap-first-drop.tex','sections/gap-first-bowl.tex','cvpr.sty']
    for name in required:
        if not (PAPER/name).is_file():
            raise SystemExit(f'Incomplete addendum; required source missing: {name}')
    files={};origins={}
    for name in required:
        destination='main.tex' if name=='gap-first-addendum.tex' else name
        raw=(PAPER/name).read_bytes()
        if destination=='main.tex':
            raw=raw.replace(b'\\newcommand{\\gfpath}{../../output/gap-first-v01}',b'\\newcommand{\\gfpath}{figures}')
        files[destination]=raw;origins[destination]=str((PAPER/name).relative_to(ROOT))
    for name in required:
        if not name.endswith('.tex'):continue
        tex=(PAPER/name).read_text()
        for dependency in re.findall(r'\\(?:input|include)\{([^}]+)\}',tex):
            local=dependency if dependency.endswith('.tex') else dependency+'.tex'
            if local not in required:
                raise ValueError(f'Unpackaged TeX dependency: {local}')
        if re.search(r'\\(?:bibliography|addbibresource)\{',tex):
            raise ValueError('External bibliography dependency requires explicit packaging')
        for target in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',tex):
            if target.startswith(r'\gfpath/'):
                relative=target[len(r'\gfpath/'):]
                source=ROOT/'output/gap-first-v01'/relative
                destination='figures/'+relative
            elif target.startswith('../../output/gap-first-v01/'):
                relative=target[len('../../output/gap-first-v01/'):]
                source=ROOT/'output/gap-first-v01'/relative
                destination='figures/'+relative
                source_name='main.tex' if name=='gap-first-addendum.tex' else name
                files[source_name]=files[source_name].replace(target.encode(),destination.encode())
            else:
                raise ValueError(f'Unrecognized figure dependency; inspect manually: {target}')
            if not source.resolve().is_relative_to((ROOT/'output/gap-first-v01').resolve()):
                raise ValueError(f'Figure escapes campaign output directory: {source}')
            if source.suffix.lower() not in {'.pdf','.png','.jpg','.jpeg'}:
                raise ValueError(f'Unexpected figure file type: {source}')
            if not source.is_file():raise FileNotFoundError(source)
            files[destination]=source.read_bytes();origins[destination]=str(source.relative_to(ROOT))
    files['build.sh']=b'#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\ntectonic -X compile main.tex\n'
    files['README.txt']=b'''Gap-first experimental addendum: editable standalone LaTeX source.

Run: sh build.sh
Requires Tectonic and its standard LaTeX packages. All included figure PDFs
embed their fonts; the original simulation environments are not required to
build this document. Experimental reproduction instructions and raw-data
policies remain in the repository reports/gap-first-v01/ directory.

This supplements the existing joint-identification manuscript; it does not
replace or relabel earlier rendering, public-data, or contact results.
'''
    rows=[dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
               original_repository_path=origins.get(name,'generated packaging metadata'))
          for name,data in sorted(files.items())]
    files['SOURCE_MANIFEST.json']=(json.dumps(rows,indent=2)+'\n').encode()
    output=PAPER/'release/gap-first-latex-source.zip';output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,11,7,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o100755 if name=='build.sh' else 0o100644)<<16
            archive.writestr(info,data)
    report=dict(source_zip=str(output.relative_to(ROOT)),files=len(files),bytes=output.stat().st_size,
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                completeness=completion,
                note='Packaging only; standalone build and visual verification are separate required release checks.')
    (ROOT/'reports/gap-first-v01/source-package.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()

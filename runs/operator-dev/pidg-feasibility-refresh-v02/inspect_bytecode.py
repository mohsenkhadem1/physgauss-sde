"""Inspect released CPython3.8 bytecode without importing/executing it."""
import argparse
import hashlib
import json
import marshal
from pathlib import Path
import types


def fingerprint(code):
    def value(item):
        if isinstance(item,types.CodeType):return {'code_fingerprint':fingerprint(item)}
        if isinstance(item,(str,bytes)):
            data=item.encode() if isinstance(item,str) else item
            return {'constant_sha256':hashlib.sha256(data).hexdigest(),'type':type(item).__name__}
        if isinstance(item,(list,tuple)):return [value(x) for x in item]
        if item is None or isinstance(item,(int,float,bool)):return item
        return {'type':type(item).__name__,'repr_sha256':hashlib.sha256(repr(item).encode()).hexdigest()}
    data={'bytecode':code.co_code.hex(),'constants':[value(c) for c in code.co_consts],
        'names':list(code.co_names),'varnames':list(code.co_varnames),'freevars':list(code.co_freevars),
        'cellvars':list(code.co_cellvars),'argcount':code.co_argcount,'kwonlyargcount':code.co_kwonlyargcount,
        'posonlyargcount':code.co_posonlyargcount,'flags':code.co_flags}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def named(code,name):
    for c in code.co_consts:
        if isinstance(c,types.CodeType):
            if c.co_name==name:return c
            result=named(c,name)
            if result is not None:return result
    return None


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pyc',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    raw=args.pyc.read_bytes();module=marshal.loads(raw[16:]);other=compile(args.source.read_text(),'<candidate-source>','exec')
    result={'execution_of_audited_code':False,'decompilation_or_reconstructed_source_saved':False,
        'released_pyc_sha256':hashlib.sha256(raw).hexdigest(),'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'fingerprint_scope':'CPython3.8 code, constants, names, arguments and flags; filenames, line tables and source line numbers excluded. No code is imported or executed.',
        'released_module_code_fingerprint':fingerprint(module),'published_module_code_fingerprint':fingerprint(other),
        'equal_entire_module_code_fingerprint':fingerprint(module)==fingerprint(other),'classes':[]}
    for name in ['VelocityNet','FourierTimeEmbed']:
        a=named(module,name);b=named(other,name)
        row={'name':name,'released_pyc_present':a is not None,'published_source_present':b is not None}
        if a is not None and b is not None:
            row.update(released_pyc_code_fingerprint=fingerprint(a),published_source_code_fingerprint=fingerprint(b),
                equal_code_fingerprint=fingerprint(a)==fingerprint(b),methods=[])
            for method in ['__init__','forward','fast_bilinear_interp']:
                ac=named(a,method);bc=named(b,method)
                if ac is not None and bc is not None:
                    row['methods'].append({'name':method,'released_argument_names':list(ac.co_varnames[:ac.co_argcount]),
                        'source_argument_names':list(bc.co_varnames[:bc.co_argcount]),'equal_code_fingerprint':fingerprint(ac)==fingerprint(bc)})
        result['classes'].append(row)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

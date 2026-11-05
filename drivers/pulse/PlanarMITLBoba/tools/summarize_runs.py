import sys,json,re,os
sys.path.insert(0, __import__('os').environ.get('MITL_EXPERIMENTS', '/work/overleaf/Planar-MITL-DPG-BoBa/experiments'))
from compare_drivers import compare
for t,v in [a.split(':') for a in sys.argv[1:]]:
    log=open(t+'.log').read(); gr=[float(x) for x in re.findall(r"global_res: ([0-9.e+-]+)",log)]
    rk=dict((b,r.replace(' ','')) for b,r in re.findall(r"^comparison block=(\S+) solution_ranks= \(([^)]*)\)",log,re.M))
    sc=dict((b,int(n)) for b,n in re.findall(r"^comparison block=(\S+) solution_ranks=.*scalars=(\d+)",log,re.M))
    if not os.path.exists(t+'_metrics.json'): print(t,'running: sweeps',len(gr),[f'{x:.1e}' for x in gr]); continue
    tm=json.load(open(t+'_metrics.json')); vm=json.load(open(v+'_metrics.json')); e=compare(v,t)
    print(f"{t:18s} e_h={e['relative_discretization_error']:.3e} rho={e['relative_truncation_error']/e['relative_discretization_error']:.3f} res={tm['mixed_residual']:.2e} sweeps={len(gr)} T_TT={tm['total_seconds']:.1f} T_C={vm['total_seconds']:.1f} test={rk.get('test')} ({sc.get('test')}) f={rk.get('f')}")

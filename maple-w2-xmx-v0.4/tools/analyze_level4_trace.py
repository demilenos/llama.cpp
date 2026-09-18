#!/usr/bin/env python3
"""Analyze SYCL timestamp intervals: sum != elapsed, and not cross-API GPU time."""
import csv,json,statistics,sys
from pathlib import Path
from collections import defaultdict

def intervals(rows):
    a=[];total=0
    for i,r in enumerate(rows):
        lo,hi=int(r['start_ns']),int(r['end_ns']);parent=int(r['parents'])
        if hi<lo or parent>>i:raise ValueError('Invalid timestamp/parent DAG')
        for j in range(i):
            if parent>>j&1 and lo<int(rows[j]['end_ns']):raise ValueError('Child before producer end')
        a.append((lo,hi));total+=hi-lo
    if not a:raise ValueError('No intervals')
    a.sort();begin=a[0][0];end=max(x[1] for x in a);lo,hi=a[0];union=0
    for x,y in a[1:]:
        if x>hi:union+=hi-lo;lo,hi=x,y
        else:hi=max(hi,y)
    union+=hi-lo
    return dict(gpu_span_us=(end-begin)/1000,kernel_sum_us=total/1000,union_busy_us=union/1000,overlap_us=(total-union)/1000,gap_us=(end-begin-union)/1000)
def main():
    p=Path(sys.argv[1]);groups=defaultdict(list)
    for r in csv.DictReader((p/'stages.csv').open()):groups[(r['repeat'],r['arm'])].append(r)
    rows=[]
    for key,rs in groups.items():
        rs.sort(key=lambda r:int(r['index']));rows.append(dict(repeat=key[0],arm=key[1],**intervals(rs)))
    with (p/'intervals.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    samples=list(csv.DictReader((p/'samples.csv').open()));summary={}
    for arm in sorted({r['arm']for r in samples}):
        values=[float(r['wall_us'])for r in samples if r['arm']==arm]
        summary[arm]={'n':len(values),'wall_median_us':statistics.median(values),'wall_min_us':min(values),'wall_max_us':max(values)}
    summary['scope']='SYCL-only segmented-host-wait diagnostic; NOT Vulkan interop savings or model TPS'
    (p/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

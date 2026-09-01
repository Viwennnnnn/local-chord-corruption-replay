#!/usr/bin/env python3
"""Audit matched rows used by the G22 real-vs-synthetic comparison."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from scipy.stats import wilcoxon

def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--synthetic',type=Path,required=True); ap.add_argument('--real',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 s={(r['track_key'],r['seed']):r for r in read(a.synthetic)}; r={(x['track_key'],x['seed']):x for x in read(a.real)}; keys=sorted(set(s)&set(r)); missing_s=sorted(set(r)-set(s)); missing_r=sorted(set(s)-set(r)); metrics={}
 for sm,rm in [('corrupted_target_effect','changed_target_effect'),('inside_output_change','inside_output_change'),('localization_contrast','localization_contrast')]:
  d=[float(s[k][sm])-float(r[k][rm]) for k in keys]; metrics[sm]={'matched_rows':len(d),'mean_synthetic_minus_real':sum(d)/len(d),'positive_rows':sum(x>0 for x in d),'wilcoxon_p':float(wilcoxon(d).pvalue)}
 out={'protocol':'g22_pairing_audit_v1','synthetic_rows':len(s),'real_rows':len(r),'matched_rows':len(keys),'tracks':len(set(k[0] for k in keys)),'seeds':sorted(set(k[1] for k in keys)),'missing_synthetic':missing_s,'missing_real':missing_r,'metrics':metrics,'pass':not missing_s and not missing_r and len(keys)==90}; a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); return 0 if out['pass'] else 2
if __name__=='__main__': raise SystemExit(main())

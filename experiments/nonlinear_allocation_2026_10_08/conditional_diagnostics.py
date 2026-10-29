"""Conditional Monte Carlo evidence for every fixed control instance."""
import argparse
import hashlib
import json
from pathlib import Path
import run_search as r
from experiment_utils import student_t_quantile,student_t_two_sided_pvalue

np,s = r.np,r.s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directories',default='confirmation,budget_extension')
    args = parser.parse_args()
    rows = []
    for dirname in args.directories.split(','):
        directory = r.HERE/dirname
        manifest = json.loads((directory/'manifest.json').read_text())
        paths = sorted((directory/'independent_evaluation').glob('*_seed*_rho*.json'))
        assert len(paths)==len(manifest['seeds'].split(','))*len(manifest['rhos'].split(','))
        for path in paths:
            c = json.loads(path.read_text())
            pb = np.asarray(c['private']['PB-AMD']['path_errors'])
            for baseline in ('LR','Decay','Linearized','Refined-geometric'):
                b = np.asarray(c['private'][baseline]['path_errors'])
                assert len(pb)==len(b)==512
                ratio = pb.mean()/b.mean()
                gain = 100*(1-ratio)
                influence = pb-ratio*b
                se = 100*np.std(influence,ddof=1)/(np.sqrt(len(pb))*b.mean())
                differences = b-pb
                dse = np.std(differences,ddof=1)/np.sqrt(len(pb))
                tstat = differences.mean()/dse
                pvalue = student_t_two_sided_pvalue(float(tstat),len(pb)-1)
                critical = student_t_quantile(.975,len(pb)-1)
                rows.append({'study':dirname,'seed':c['seed'],'rho':c['rho'],'baseline':baseline,
                    'error_reduction_percent':float(gain),'Monte_Carlo_standard_error':float(se),
                    'accuracy_difference_pp':100*(c['private']['PB-AMD']['accuracy']-c['private'][baseline]['accuracy']),
                    'conditional_CI95_low':float(gain-critical*se),'conditional_CI95_high':float(gain+critical*se),
                    'paired_path_pvalue':float(pvalue),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    family = len(rows)
    critical = student_t_quantile(1-.05/(2*family),511)
    for row in rows:
        row['bonferroni_pvalue_all_instances'] = min(1.,family*row['paired_path_pvalue'])
        row['simultaneous_MC_low'] = row['error_reduction_percent']-critical*row['Monte_Carlo_standard_error']
        row['simultaneous_MC_high'] = row['error_reduction_percent']+critical*row['Monte_Carlo_standard_error']
    report = {'scope':'Post hoc conditional Monte Carlo diagnostics, holding each dataset split, initialization, minibatch sequence and selected schedule fixed. The 512 Gaussian trajectories estimate the objective of that one control problem; they are not 512 independent training-problem replicates.',
        'method':'Ratio-of-means improvement with paired delta-method t intervals; paired raw-error-difference t diagnostics and Bonferroni correction over every listed instance-by-baseline comparison. Intervals are Monte Carlo approximations, not finite-sample exact statements.',
        'cohort_inference':'Use the separate seed-level summaries for conclusions across random training problems.',
        'family_size':family,'comparisons':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target = r.HERE/('conditional_diagnostics.json' if ',' in args.directories else args.directories+'/conditional_diagnostics.json')
    s.save(target,report)
    print(json.dumps([row for row in rows if row['baseline']=='Linearized'],indent=2))


if __name__=='__main__':
    main()

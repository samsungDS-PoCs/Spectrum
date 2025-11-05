import argparse
import numpy as np
import os

def read_file(infile, result_dict):
    for line in open(infile):
        if line.startswith('#'):
            continue
        if 'id,' in line or 'ID,' in line:
            continue
        linesp = line.strip().split(',')
        name = linesp[0]
        result_dict[name] = (np.array(linesp[1].split(), dtype=np.float64), np.array(linesp[2].split(), dtype=np.float64))


def fn_kl(p, q):
    return np.sum(p * np.log(p / q))
def fn_kl2(p,q):
    return np.sum(p * np.log2(p / q))

def compute_sid(p: np.ndarray, q: np.ndarray, x = None, threshold=0.05, epsilon=1e-12):
    p = np.maximum(p - threshold, 0.0)
    q = np.maximum(q - threshold, 0.0)

    p = p / np.sum(p)  + epsilon
    q = q / np.sum(q)  + epsilon
    m = p+q
    m = m / np.sum(m)

    kl_pq = fn_kl(p, q) 
    kl_qp = fn_kl(q, p) 

    sid_value = kl_pq + kl_qp #SID = KL(p‖q) + KL(q‖p)
    sis_value = 1.0 / (sid_value + 1.0)
    jsd = fn_kl2(p, m) + fn_kl2(q, m)
    
    cp = np.cumsum(p)
    cq = np.cumsum(q)
    dx = np.diff(x)
    emd = np.sum(np.abs(cp[:-1] - cq[:-1])*dx)

    return sid_value, sis_value, jsd, emd
        
def main():
    parser = argparse.ArgumentParser(description="python --exp-file {exp_file} --pred-file {pred_file}")
    parser.add_argument("--exp-file", type=str, default='IrDB/raw/y_spec_exp.csv')
    parser.add_argument("--pred-file", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.exp_file):
        args.exp_file = '../IrDB/raw/y_spec_exp.csv'
    if not os.path.exists(args.exp_file):
        raise Exception("Please check file path of {exp_file} (y_spec_exp.csv)")
    if not os.path.exists(args.pred_file):
        raise Exception("Please check file path of {pred_file} (p_spec.csv)")

    pred_dict = {}
    exp_dict = {}
    read_file(args.pred_file, pred_dict)
    read_file(args.exp_file, exp_dict)
    id_list = pred_dict.keys() & exp_dict.keys()
            
    print('#id,SID,SIS,JSD,EMD')
    
    avg_sid, avg_sis, avg_jsd, avg_emd = 0.0, 0.0, 0.0, 0.0
    n_mol = 0.0
    for name in id_list:
        sid, sis, jsd, emd = compute_sid(exp_dict[name][1], pred_dict[name][1], x=exp_dict[name][0])
        avg_sid += sid
        avg_sis += sis
        avg_jsd += jsd
        avg_emd += emd
        n_mol += 1.0
        print(f'{name},{sid},{sis},{jsd},{emd}')
    avg_sid, avg_sis = avg_sid / n_mol, avg_sis / n_mol
    avg_jsd, avg_emd = avg_jsd / n_mol, avg_emd / n_mol
    print(f'#AVG,{avg_sid},{avg_sis},{avg_jsd},{avg_emd}')


if __name__ == '__main__':
    main()

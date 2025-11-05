import argparse
import numpy as np
import os
from calc_peak_fwhm import get_FWHM
ini = 25
fin = 75
header = ['id'] + [f'{(i*0.01):.2f}' for i in range(ini, fin+1)]
header = ','.join(header)

def list2line(li):
    w_line = ''
    for i_aa, aa in enumerate(li):
        w_line += '%.6f '%aa
    return w_line[:-1]

def get_fwxm(infile, outfile):
    data_list = [header]
    for line in open(infile):
        if 'id,' in line or 'ID,' in line:
            continue
        linesp = line.strip().split(',')
        name = linesp[0]
        spec_x = np.array(linesp[1].split(), dtype=np.float64)
        inten_s = np.array(linesp[2].split(), dtype=np.float64)
        
        w_list = []
        for i in range(ini, fin+1):
            half_value = 0.01*i
            fwhm = get_FWHM(spec_x, inten_s, half_value=half_value)
            w_list.append(str(fwhm))

        w_line = f'{name},' + ','.join(w_list)
        data_list.append(w_line)
    
    w_file = open(outfile, 'w')
    for line in data_list:
        w_file.write('%s\n'%line)
    w_file.close()

def main():
    parser = argparse.ArgumentParser(description="python --input-file {spectrum_file} --output-file {fwhm_file}")
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise Exception("Please check file path of {input_file} (y_spec_exp.csv or p_spec.csv)")

    if args.output_file == None or (not os.path.exists(args.output_file)):
        if '.csv' in args.input_file:
            args.output_file = args.input_file.replace('.csv', '_FWXM.csv')
        else:
            args.output_file = args.input_file + '_FWXM.csv'
 
    get_fwxm(args.input_file, args.output_file)
    print (f"The FWXM of the spectrum in the {args.input_file} were calculated and saved to the {args.output_file}")
 
if __name__ == '__main__':
    main()

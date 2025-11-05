import argparse
import numpy as np
import os
import scipy.interpolate as spi
targets = ['id', 'Peak', 'FWHM', 'PL_center', 'FWQM']
tar_line = ','.join(targets)

def fn_interpolate(x,y, tar, k=1):
    ipo = spi.splrep(x,y,k=k)
    return spi.splev(tar, ipo)

def find_half_point(spec_x, intensity, half_value=0.5):
    for i, inte in enumerate(intensity):
        if inte > half_value:
            break
    if (i == 0) or (i == len(intensity) -1):
        return spec_x[i]
    else:
        evs = spec_x[i-1:i+1]
        int05 = intensity[i-1:i+1]
        if int05[0] > int05[1]:
            int05 = np.flip(int05)
            evs = np.flip(evs)
        return fn_interpolate(int05, evs, half_value)

def get_PEAK(spec_x, intensities):
    i_max = 0
    peak_max = intensities[i_max]
    for i in range(1, len(spec_x)):
        if intensities[i] > peak_max:
            peak_max = intensities[i]
            i_max = i
    return spec_x[i_max]

def get_PEAK2(spec_x, spec_y, cut_val=0.98):
    max_peak = None
    i98 = np.where(spec_y>cut_val)[0]
    if np.size(i98) < 1:
        return max_peak
    
    x98 = spec_x[i98]
    y98 = spec_y[i98] - cut_val
    max_peak = np.sum(x98*y98) / np.sum(y98)
    return max_peak

def get_FWHM(spec_x, intensities, half_value=0.5):
    b_spec_x = np.flip(spec_x)
    b_intensities = np.flip(intensities)
    rhalf = find_half_point(spec_x, intensities, half_value=half_value)
    lhalf = find_half_point(b_spec_x, b_intensities, half_value=half_value)
    fwhm = abs(rhalf-lhalf)

    return fwhm

def get_peak(infile, outfile):
    data_list = []
    data_list.append(tar_line)
    for line in open(infile):
        if 'id,' in line:
            continue
        linesp = line.strip().split(',')
        name = linesp[0]
        spec_x = np.array(linesp[1].split(), dtype=np.float64)
        inten_s = np.array(linesp[2].split(), dtype=np.float64)
        peak_nm = get_PEAK(spec_x, inten_s)
        fwhm_nm = get_FWHM(spec_x, inten_s)
        fwqm_nm = get_FWHM(spec_x, inten_s, half_value=0.25)
        pl_center = get_PEAK2(spec_x, inten_s, cut_val=0.0)
        w_line = f'{name},{peak_nm},{fwhm_nm},{pl_center},{fwqm_nm}'
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
            args.output_file = args.input_file.replace('.csv', '_peak_fwhm.csv')
        else:
            args.output_file = args.input_file + '_peak_fwhm.csv'
    
    get_peak(args.input_file, args.output_file)
    
    print (f"The peaks and FWHMs of the spectrum in the {args.input_file} were calculated and saved to the {args.output_file}")

if __name__ == '__main__':
    main()

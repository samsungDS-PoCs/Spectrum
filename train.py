import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Spectrum prediction based on physics-informed neural network")
    parser.add_argument(
        "--spectrum-type",
        type=str,
        default='FC',
        choices=["Naive", "GMM", "FC"],
        help="FC: Spectrum prediction based on Franck-Condon progression. GMM: Spectrum prediction based on Gaussian Mixture-Model Naive: Not consider Spectrum Loss.",
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        choices=["Geoformer", "PaiNN", "Equiformer"],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    return parser.parse_args()

base_models = ["Geoformer", "PaiNN", "Equiformer"]
spectrum_types = ["Naive", "GMM", "FC"]

def main():
    i_seed = 0
    i_fold = 0
    args = get_args()
    if args.base_model == None or args.spectrum_type == None:
        print ("python train.py --base-model {Geoformer|PaiNN|Equiformer} [--spectrum-type {Naive|GMM|FC} (default:FC)] [--batch-size <int> (default:16)]")
        print ("Please select base-model from Geoformer, PaiNN, or Equiformer")
        print ("Please select spectrum-type from Naive, GMM, or FC")
        return
    batch_size = args.batch_size

    if not args.base_model in base_models:
        raise Exception("Undefined model. Please select one from Geoformer, PaiNN, or Equiformer.")
    if not args.spectrum_type in spectrum_types:
        raise Exception("Undefined spectrum_type. Please select one from Naive, GMM, or FC.")

    train_file = f'train_{args.base_model}'
    split_npz = f'IrDB/raw/CV811/splits.{i_seed}.{i_fold}.npz'
    log_path = f'results_{args.base_model}/{i_seed}/{i_fold}'
    
    if args.base_model == 'Geoformer':
        cmd_line = f'python -m train_Geoformer --conf geoformer/examples/{args.spectrum_type}.yml --log-dir {log_path} --seed {i_seed} --splits {split_npz} --batch-size {batch_size}'
    elif args.base_model == 'PaiNN':
        cmd_line = f'python -m train_PaiNN --spectrum-type {args.spectrum_type} --output-dir {log_path} --split-index-npz {split_npz} --seed {i_seed} --batch-size {batch_size}'
    elif args.base_model == 'Equiformer':
        cmd_line = f'python -m train_Equiformer --spectrum-type {args.spectrum_type} --output-dir {log_path} --split-index-npz {split_npz} --seed {i_seed} --batch-size {batch_size}'
    
    # os.system(f'phd run -p mai_small_gpu -ng 1 -GR "name==H100" -- {cmd_line}')
    os.system(f'{cmd_line}')

    
if __name__ == "__main__":
    main()
    

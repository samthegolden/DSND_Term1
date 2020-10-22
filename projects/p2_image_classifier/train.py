from argparse import ArgumentParser


if __name__ == "__main__":

    ap = ArgumentParser(description="ML")
    ap.add_argument("data_directory", help="The data directory that feeds the script.")
    ap.add_argument('--save_dir', help="Save data to dir.")
    ap.add_argument('--arch', help="Architecture.")
    ap.add_argument("--learning_rate", type=float, help="Learning rate")
    ap.add_argument('--hidden_units', type=int, help="Hidden units")
    ap.add_argument('--epochs', type=int, help="Epochs")
    ap.add_argument('--gpu', action='store_true', help="GPU")
    
    args = ap.parse_args()
    print(args.data_directory)
    print(args.save_dir)
    print(args.arch)
    print(args.learning_rate)
    print(args.hidden_units)
    print(args.epochs)
    print(args.gpu)
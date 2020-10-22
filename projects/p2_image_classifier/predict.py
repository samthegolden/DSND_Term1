from argparse import ArgumentParser


if __name__ == "__main__":

    ap = ArgumentParser(description="ML")
    ap.add_argument("path_to_image", help="The path to a single image.")
    ap.add_argument('checkpoint', help="Checkpoint file.")
    ap.add_argument('--top_k', type=int, help="Top K classes.")
    ap.add_argument("--category_names", type=str, help="Json mapping of categories to real names")
    ap.add_argument('--gpu', action='store_true', help="GPU.")
    
    args = ap.parse_args()
    print(args.path_to_image)
    print(args.checkpoint)
    print(args.top_k)
    print(args.category_names)
    print(args.gpu)
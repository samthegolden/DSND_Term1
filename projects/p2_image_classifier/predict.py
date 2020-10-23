from argparse import ArgumentParser
import torch

def load_checkpoint(filepath):
    '''
    checkpoint = torch.load(filepath)
    print(checkpoint.keys())
    ### model = models.vgg16(pretrained=True)
    ## How do I instantiate 'model'?
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    '''
    model = torch.load(filepath)
    return model

def predict(path_to_image, checkpoint, top_k=5, category_names='cat_to_name.json', gpu=True):
    model = load_checkpoint(checkpoint)

    print(f"CLASSES TO IDX:", model.classes_to_idx)


if __name__ == "__main__":

    ap = ArgumentParser(description="ML")
    ap.add_argument("path_to_image", help="The path to a single image.")
    ap.add_argument('checkpoint', help="Checkpoint file.")
    ap.add_argument('--top_k', type=int, help="Top K classes.")
    ap.add_argument("--category_names", type=str, help="Json mapping of categories to real names")
    ap.add_argument('--gpu', action='store_true', help="GPU.")
    
    args = ap.parse_args()
    
    '''
    print(args.path_to_image)
    print(args.checkpoint)
    print(args.top_k)
    print(args.category_names)
    print(args.gpu)
    '''

    kwargs = dict(top_k=args.top_k, category_names=args.category_names, gpu=args.gpu)
    # only pass parameters that are not None
    predict(args.path_to_image, args.checkpoint, **{k: v for k, v in kwargs.items() if v is not None})

    ## RUN

    # python predict.py 'flowers/test/1/image_06743.jpg' '.ipynb_checkpoints/Image Classifier Project-checkpoint.ipynb'
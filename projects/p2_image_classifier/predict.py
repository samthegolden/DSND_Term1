from argparse import ArgumentParser
import torch
from torchvision import datasets, transforms, models
from math import floor
import numpy as np
from numpy import ndarray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Write a function that loads a checkpoint and rebuilds the model
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
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained = True) 
       
    model.classifier = checkpoint['classifier']
    model.load_state_dict (checkpoint['state_dict'])
    # getting the mapping
    model.class_to_idx = checkpoint['mapping']
    
    return model

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
    import PIL
    from PIL import Image
    
    '''
    im = Image.open(image)
    #im.rotate(45).show()
    #im = im.resize((256, 256)) # i'm not keeping the aspect ratio...
    im = im.thumbnail([256, 256],PIL.Image.ANTIALIAS) 
    print(f" IM SIZE: {im.size}")
    im = im.crop((224, 224,224,224))
    '''
    
    image = Image.open(image)
    width, height = image.size
    size = 256, 256
    ratio = float(width) / float(height)
    
    if width > height:
        newheight = ratio * size[0]
        image = image.resize((size[0], int(floor(newheight))), Image.ANTIALIAS)
    else:
        newwidth = size[1] / ratio
        image = image.resize((int(floor(newwidth)), size[1]), Image.ANTIALIAS)
    size = image.size
    print(f"SIZE: {size}")
    image.show()
    # See this for better understanding of crop() -> https://stackoverflow.com/questions/20361444/cropping-an-image-with-python-pillow
    # It starts counting on top left corner... not in the center
    image = image.crop((
        size[0] / 2 - (224/2),
        size[1] / 2 - (224/2),
        size[0] / 2 + (224/2),
        size[1] / 2 + (224/2),
    ))
    
    print(size[0] / 2 - (224/2))
    print(f"SIZE AFTER CROP: {image.size}")

    # convert colors
    np_image = np.array(image)
    im = np_image.astype('float64') / 255 # convert to [0,1]
    
    # normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im = (im - mean) / std
    
    im = im.transpose((2,0,1)) # The color channel needs to be first and retain the order of the other two dimensions.
    
    return im


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, checkpoint, top_k=5, category_names='cat_to_name.json', gpu=True):
    model = load_checkpoint(checkpoint)

    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    import json

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    image = process_image(image_path)
    #image = imshow(image)
    
    image = torch.from_numpy(image)
    
    torch_tensor = torch.tensor(image).float() # Convert the image to float
    # Make it as batch of a single image
    torch_tensor = torch_tensor.unsqueeze(0) # torch.Size([1, 3, 224, 224]) (the model expects a 32 image batch) it adds extra dimension
    image = torch_tensor.to(device) # without it there will be error RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'weight'
    
    # Move the image tensor to the same device as that of model, then get prediction
    # TURN OFF GRADIENT CONPUTATION!
    with torch.no_grad():
        ps = torch.exp(model(image)) # need to convert to pytorch tensor...    print(ps.shape)
    
    probs, indeces = ps.topk(topk)
    if device == 'cpu':
        probs = probs.numpy()
        indeces = indeces.numpy()
    else:
        probs = probs.cpu().numpy()
        indeces = indeces.cpu().numpy()
        
    print("MODEL:", model)
    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[i] for i in indeces[0].tolist()]

    class_names = [cat_to_name[c] for c in classes]
    
    return probs, class_names

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

    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    kwargs = dict(top_k=args.top_k, category_names=args.category_names, gpu=args.gpu)
    # only pass parameters that are not None
    ps, classe_names = predict(args.path_to_image, args.checkpoint, **{k: v for k, v in kwargs.items() if v is not None})

    print(ps, class_names)

    ## RUN

    # python predict.py 'flowers/test/1/image_06743.jpg' '.ipynb_checkpoints/Image Classifier Project-checkpoint.ipynb'
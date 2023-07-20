from unet import UNet
import torch
from processLIDC2 import Patient
import matplotlib.pyplot as plt
import torch.nn as nn
from torchviz import make_dot

if __name__ =='__main__':
    model = UNet(in_channels=1, out_channels=1, init_features=32) # , dropout_rate=0.2)
    
    patient = Patient('LIDC-IDRI-0002')
    
    patient.scale()
    
    images, mask = patient.get_tensors(scaled=True)
    
    print(images.shape)
    print(mask.shape)
    
    pred = model(images[30:40,:,:])
    prediccion = pred.cpu().detach().numpy()[0,0,:,:]
    plt.imshow(prediccion)
    plt.show()
    print(prediccion.shape)
    
    # graph = make_dot(pred, params=dict(model.named_parameters()))

    # # Guarda la imagen
    # graph.format = 'png'  # También puedes guardarla en otros formatos como 'svg'
    # graph.render("unet_model_graph")
    
    
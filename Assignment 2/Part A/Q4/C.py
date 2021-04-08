from PIL import Image
import matplotlib.pyplot as plt
import keras
import numpy as np
%matplotlib inline

img_PIL = Image.open(r'/content/gdrive/MyDrive/inaturalist_12K/val/Aves/049650eac0f12d7a16c785a0f1e06e0f.jpg')
#plt.imshow(img_PIL)

img = img_PIL.resize((224, 224))
img= keras.preprocessing.image.img_to_array(img)
#img.shape

model = keras.models.load_model('Best model Part A.h5')

filters, biases = model.layers[0].get_weights()

def forward_pass(img,f,b) :
    
    h_f, w_f, d_f = f.shape
    h_out, w_out = img.shape[0]-f.shape[0]+1,img.shape[1]-f.shape[1]+1
    
    
    output = np.zeros((218,218))
    
    for i in range(h_out):
        for j in range(w_out):
            h_start, w_start = i, j
            h_end, w_end = h_start + h_f, w_start + w_f
            output[i, j] = np.sum(img[h_start:h_end, w_start:w_end,:] *f )

    return output + b



plt.rcParams["figure.figsize"] = (30,30)
for i in range(64):
  f = filters[:, :, :, i]
  b=biases[i]
  plt.subplot(8,8, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.title(f'Filter : {i+1}',fontsize=15,fontweight='bold')
  plt.imshow(forward_pass(array1,f,b))
  
 
		

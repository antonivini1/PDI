from PIL import Image
import numpy as np
import math

def split_channels(img):
    Red = np.asarray(img)
    Red[:,:,1] *= 0
    Red[:,:,2] *= 0

    Green = np.asarray(img)
    Green[:,:,0] *= 0
    Green[:,:,2] *= 0

    Blue = np.asarray(img)
    Blue[:,:,0] *= 0
    Blue[:,:,1] *= 0
    
    return Image.fromarray(Red), Image.fromarray(Green), Image.fromarray(Blue)
def which_c(number):
    if(number == 0):
        return math.sqrt(0.5)
    else:
        return 1
def DCT(x,N):
    X = np.zeros(N)
    for k in range(N):
        c = which_c(k)
        summ = 0
        for n in range(N):
            summ += x[n] * math.cos(((2*n+1)*math.pi*k)/(2*N))
        X[k] = math.sqrt(2/N) * c * summ
    return X
def iDCT(X,N):
    x = np.zeros(N)
    for n in range(N):
        summ = 0
        for k in range(N):
            c = which_c(k)
            summ += X[k] * c * math.cos(((2*n+1)*math.pi*k)/(2*N))
        x[n] = math.sqrt(2/N) * summ
    return x

img_src = Image.open("lena256.png")
img_arr = np.asarray(img_src)
R,C = img_arr.shape
X = np.zeros((R,C))
x = np.zeros((R,C))
img_src.show()

for k in range(R):
    X[k,:] = DCT(img_arr[k,:],R)
for l in range(C):
    X[:,l] = DCT(X[:,l],C)
#X /= X[0][0]

#img_dct = np.log(np.abs(X)+1)
img_dct = X * (255.0/X.max())
img_dct = Image.fromarray(img_dct)
img_dct = img_dct.convert("P")
img_dct.show()

for m in range(R):
    x[m,:] = iDCT(X[m,:],R)
for n in range(C):
    x[:,n] = iDCT(x[:,n],C)

img_idct = x * (255.0/x.max())
img_idct = Image.fromarray(img_idct)
img_idct = img_idct.convert("P")
img_idct.show()

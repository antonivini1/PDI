from PIL import Image
from scipy.io import wavfile
from numba import jit
import numpy as np
import math

@jit
def DCT(x,N):
    X = np.zeros(N)
    for k in range(N):
        c = (math.sqrt(0.5) if (k == 0) else 1)
        summ = 0
        for n in range(N):
            summ += x[n] * math.cos(((2*n+1)*math.pi*k)/(2*N))
        X[k] = math.sqrt(2/N) * c * summ
    return X
@jit
def iDCT(X,N):
    x = np.zeros(N)
    for n in range(N):
        summ = 0
        for k in range(N):
            c = (math.sqrt(0.5) if (k == 0) else 1)
            summ += X[k] * c * math.cos(((2*n+1)*math.pi*k)/(2*N))
        x[n] = math.sqrt(2/N) * summ
    return x
def Passa_Baixas(x,f_c):
    for k in range(len(x)):
        if(x[k] > f_c):
            x[k] = 0
    return x
def Graves(g, k, f_c, n):
    Y = (g / (math.sqrt(1 + (k/f_c)**(2*n)))) + 1
    return Y

def quest1(img_name):
    img_src = Image.open(img_name)
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

    #X = np.log(np.abs(X)+1)
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
def main():
    quest1("lena256.png")
    # snd_sample, snd_data = wavfile.read("MasEstamosAiPraMais.wav")
    # #snd_data = snd_data.astype(np.float64) / 2**15
    
    # snd_dct = DCT(snd_data,len(snd_data))
    # #snd_dct_filtered = Passa_Baixas(snd_dct,12520)
    # snd_dct_boosted = np.zeros(snd_dct.shape)
    # for k in range(len(snd_dct)):
    #     snd_dct_boosted[k] = Graves(0.5, k, 12520, 2) * snd_dct[k]
    # snd_write = iDCT(snd_dct_boosted,len(snd_dct_boosted))
    
    # wavfile.write("example.wav", snd_sample, snd_write)

if __name__== "__main__" :  
  main()

from PIL import Image
from scipy.io import wavfile
from numba import jit
import numpy as np
import math

@jit
def DCT(x):
    N = len(x)
    X = np.zeros(N)
    for k in range(N):
        c = (math.sqrt(0.5) if (k == 0) else 1)
        summ = 0
        for n in range(N):
            summ += x[n] * math.cos(((2*n+1)*math.pi*k)/(2*N))
        X[k] = math.sqrt(2/N) * c * summ
    return X
@jit
def DCT2d(x):
    R,C = x.shape
    X = np.zeros((R,C))

    for i in range(R):
        X[i,:] = DCT(x[i,:])
    for j in range(C):
        X[:,j] = DCT(X[:,j])
    
    return X
@jit
def iDCT(X):
    N = len(X)
    x = np.zeros(N)
    for n in range(N):
        summ = 0
        for k in range(N):
            c = (math.sqrt(0.5) if (k == 0) else 1)
            summ += X[k] * c * math.cos(((2*n+1)*math.pi*k)/(2*N))
        x[n] = math.sqrt(2/N) * summ
    return x
@jit
def iDCT2d(X):
    R,C = X.shape
    x = np.zeros((R,C))

    for i in range(R):
        x[i,:] = iDCT(X[i,:])
    for j in range(C):
        x[:,j] = iDCT(x[:,j])
    
    return x

def Graves(g, k, f_c, n):
    Y = (g / (math.sqrt(1 + (k/f_c)**(2*n)))) + 1
    return Y
@jit
def DCT_audio(x):
    N = len(x)
    X = np.zeros(N)
    for k in range(N):
        c = (math.sqrt(0.5) if (k == 0) else 1)
        Y = Graves(0.5,k,25000,2)
        summ = 0
        for n in range(N):
            summ += x[n] * math.cos(((2*n+1)*math.pi*(k*Y)/(2*N)))
        X[k] = math.sqrt(2/N) * c * summ
    return X
def Passa_Baixas(x,f_c):
    for k in range(len(x)):
        if(x[k] > f_c):
            x[k] = 0
    return x

def quest1(img_name,AC):
    print("Abrindo imagem " + img_name,end=".\n")
    img_src = Image.open(img_name)
    img_arr = np.asarray(img_src)

    print("Aplicando DCT2D na imagem.")
    X = DCT2d(img_arr)
    
    #img_dct = np.log(np.abs(X)+1)
    img_dct = X.copy()
    img_dct[0][0] = 0
    img_dct *= (255/img_dct.max())
    #img_dct = np.abs(img_dct)
    img_dct_out = Image.fromarray(img_dct)
    img_dct_out = img_dct_out.convert("P")
    img_dct_out.save("Result_DCT.png")

    print("Aplicando DCT2D inversa na imagem.")
    x = iDCT2d(img_dct)

    #x /= x[0][0]
    img_idct = x.copy()
    img_idct *= (255/img_idct.max())
    #img_idct[0][0] = 0
    img_idct_out = Image.fromarray(img_idct)
    img_idct_out = img_idct_out.convert("P")
    img_idct_out.save("Result_iDCT.png")

    print("1.2) Encontrar e exibir uma aproximação de I obtida preservando o coeficiente DC e os n coeficientes AC mais importantes de I, e zerando os demais.")
    img_dct_apx = X.flatten()
    aproximation = img_dct_apx[1:]
   
    indexes_sorted = np.abs(aproximation)
    indexes_sorted = np.argsort(indexes_sorted)
    indexes_zeroed = indexes_sorted[:(len(img_dct_apx)-AC)]
    
    aproximation[indexes_zeroed] = 0
    img_dct_apx[1:] = aproximation
    img_dct_apx = img_dct_apx.reshape((X.shape))
    img_dct_apx *= (255/img_dct_apx.max())
    img_dct_apx_out = Image.fromarray(img_dct_apx)
    img_dct_apx_out = img_dct_apx_out.convert("P")
    img_dct_apx_out.save("Result_DCT_Aproximation.png")

    img_idct_apx = iDCT2d(img_dct_apx)
    img_idct_apx *= (255/img_idct_apx.max())
    img_idct_apx_out = Image.fromarray(img_idct_apx)
    img_idct_apx_out = img_idct_apx_out.convert("P")
    img_idct_apx_out.save("Result_iDCT_Aproximation.png")

    # img_src.show()
    # Image.fromarray(X).convert("P").show()
    # print("1.1) Modulo normalizado da DCT sem o nivel DC.")
    # img_dct_out.show()
    # print("1.1) Valor do nivel DC: ", img_dct[0][0], end=".\n")
    # Image.fromarray(x).convert("P").show()
    # print("Lena normalizada sem o nivel DC.")
    # img_idct_out.show()
    # print("Aproximação da DCT de I.")
    # img_dct_apx_out.show()
    # print("Aproximação da iDCT de I.")
    # img_idct_apx_out.show()

def main():
    quest1("lena256.png",100)
    # snd_sample, snd_data = wavfile.read("MasEstamosAiPraMais.wav")
    # #snd_data = snd_data.astype(np.float64) / 2**15
    
    # snd_dct_boosted = DCT_audio(snd_data)
    # #snd_dct_filtered = Passa_Baixas(snd_dct,12520)
    # snd_write = iDCT(snd_dct_boosted,len(snd_dct_boosted))
    
    # wavfile.write("example.wav", snd_sample, snd_write)

if __name__== "__main__" :  
  main()

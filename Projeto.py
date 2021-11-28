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
    R,C = img_arr.shape
    X = np.zeros((R,C))
    x = np.zeros((R,C))
    img_src.show()

    print("Aplicando DCT2D na imagem.")
    for k in range(R):
        X[k,:] = DCT(img_arr[k,:])
    for l in range(C):
        X[:,l] = DCT(X[:,l])

    #img_dct = np.log(np.abs(X)+1)
    img_dct = X
    img_dct[0][0] = 0
    img_dct *= (255/img_dct.max())
    #img_dct = np.abs(img_dct)
    img_dct_out = Image.fromarray(img_dct)
    img_dct_out = img_dct_out.convert("P")
    img_dct_out.save("Result_DCT.png")

    print("1.1) Modulo normalizado da DCT sem o nivel DC.")
    img_dct_out.show()
    print("1.1) Valor do nivel DC: ", img_dct[0][0], end=".\n")

    print("Aplicando DCT2D inversa na imagem.")
    for m in range(R):
        x[m,:] = iDCT(img_dct[m,:])
    for n in range(C):
        x[:,n] = iDCT(x[:,n])

    #x /= x[0][0]
    img_idct = x
    img_idct *= (255.0/img_idct.max())
    #img_idct[0][0] = 0
    img_idct_out = Image.fromarray(img_idct)
    img_idct_out = img_idct_out.convert("P")
    img_idct_out.save("Result_iDCT.png")

    print("Lena normalizada sem o nivel DC.")
    img_idct_out.show()

    print("1.2) Encontrar e exibir uma aproximação de I obtida preservando o coeficiente DC e os n coeficientes AC mais importantes de I, e zerando os demais.")
    aproximation = (X.flatten())[1:]
    indexes_zero = np.abs(aproximation)



   # img_apx = X.flatten()


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

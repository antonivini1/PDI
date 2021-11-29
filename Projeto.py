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
def iDCT2d(X):
    R,C = X.shape
    x = np.zeros((R,C))

    for i in range(R):
        x[i,:] = iDCT(X[i,:])
    for j in range(C):
        x[:,j] = iDCT(x[:,j])
    
    return x

@jit
def DCT_audio(x, filtro = None):
    N = len(x)
    X = np.zeros(N)
    for k in range(N):
        c = (math.sqrt(0.5) if (k == 0) else 1)
        summ = 0
        Y = (1 if filtro == None else filtro[k])
        for n in range(N):
            summ += x[n] * math.cos(((2*n+1)*math.pi*(k*Y)/(2*N)))
        X[k] = math.sqrt(2/N) * c * summ
    return X
def Reforço_Graves(nsamples, g, f_c, n):
    Y = np.zeros(nsamples)
    for k in range(nsamples):
        Y[k] = (g / (math.sqrt(1 + math.pow(k/f_c, 2 * n)))) + 1
    return Y

def quest1(img_name,AC):
    print("1.1) Exibir o módulo normalizado da DCT de I, sem o nível DC, e o valor (numérico) do nível DC.")
    print("Abrindo imagem " + img_name,end=".\n")
    img_src = Image.open(img_name)
    img_arr = np.asarray(img_src)
    img_src.show("Imagem Original.png")

    print("Aplicando DCT2D na imagem.")
    X = DCT2d(img_arr)
    print("Imagem no domininio da frequencia.")
    Image.fromarray(X).convert("P").show("Imagem no domininio da frequencia.png")
    
    print("Normalizando, zerando o nivel DC e pegando o modulo normalizado da imagem DCT.")
    #img_dct = np.log(np.abs(X)+1)
    img_dct = X.copy()
    img_dct[0][0] = 0
    img_dct *= (255/img_dct.max())
    img_dct = np.abs(img_dct)
    img_dct_out = Image.fromarray(img_dct)
    img_dct_out = img_dct_out.convert("P")
    img_dct_out.save("Resultados/Result_DCT.png")

    print("1.1) Modulo normalizado da DCT sem o nivel DC.")
    img_dct_out.show("Modulo normalizado da DCT sem o nivel DC.png")
    print("1.1) Valor do nivel DC: ", X[0][0], end=".\n")

    print("Aplicando DCT2D inversa na imagem.")
    x = iDCT2d(img_dct)
    print("Imagem no domininio do tempo.")
    Image.fromarray(x).convert("P").show("Imagem no domininio do tempo.png")

    print("Normalizando a imagem iDCT.")
    #x /= x[0][0]
    img_idct = x.copy()
    img_idct *= (255/img_idct.max())
    #img_idct[0][0] = 0
    img_idct_out = Image.fromarray(img_idct)
    img_idct_out = img_idct_out.convert("P")
    img_idct_out.save("Resultados/Result_iDCT.png")

    print("Imagem da Lena normalizada sem o nivel DC.")
    img_idct_out.show("Imagem da Lena normalizada sem o nivel DC.png")

    print("#--------------------------------------------------------------------#")

    print("1.2) Encontrar e exibir uma aproximação de I obtida preservando o coeficiente DC e os n coeficientes AC mais importantes de I, e zerando os demais.")
    print("Copiando a imagem dct e transformando em um array 1D.")
    img_dct_apx = X.flatten()
    print("Organizando os modulos dos elementos do array e pegando os n ACs mais importantes e zerando o resto.")
    aproximation = img_dct_apx[1:]
   
    indexes_sorted = np.abs(aproximation)
    indexes_sorted = np.argsort(indexes_sorted)
    indexes_zeroed = indexes_sorted[:(len(img_dct_apx)-AC)]
    aproximation[indexes_zeroed] = 0

    print("Normalizando a imagem aproximada.")
    img_dct_apx[1:] = aproximation
    img_dct_apx = img_dct_apx.reshape((X.shape))
    img_dct_apx *= (255/img_dct_apx.max())
    img_dct_apx_out = Image.fromarray(img_dct_apx)
    img_dct_apx_out = img_dct_apx_out.convert("P")
    img_dct_apx_out.save("Resultados/Result_DCT_Aproximation.png")

    print("Aproximação da imagem DCT de I.")
    img_dct_apx_out.show("Aproximação da imagem DCT de I.png")

    print("Calculando a iDCT da imagem aproximada de I e normalizando-a.")
    img_idct_apx = iDCT2d(img_dct_apx)
    img_idct_apx *= (255/img_idct_apx.max())
    img_idct_apx_out = Image.fromarray(img_idct_apx)
    img_idct_apx_out = img_idct_apx_out.convert("P")
    img_idct_apx_out.save("Resultados/Result_iDCT_Aproximation.png")

    print("Aproximação da imagem iDCT de I.")
    img_idct_apx_out.show("Aproximação da imagem iDCT de I.")
def quest2(snd_name,g,f_c,n):
    print("2) Desenvolva um programa para reforçar os graves, no domínio DCT, de um sinal s, em formato .wav, com N amostras.")
    print("Abrindo o audio " + snd_name,end=".\n")
    snd_sample, snd_data = wavfile.read(snd_name)
    print("Convertendo o audio para float e normalizando-o.")
    snd_data_float = snd_data.astype(np.float64) / 2**15

    print("Calculando o reforço, aplicando a DCT e a iDCT.")
    Y = Reforço_Graves(len(snd_data_float),g,f_c,n)
    snd_dct_boosted = DCT_audio(snd_data_float, Y)
    snd_write = iDCT(snd_dct_boosted)

    print("Desnormalizando e reconvertendo o audio para int.")
    snd_write_int = snd_write * 2**15
    snd_write_int = snd_write_int.astype(np.int16)

    wavfile.write("Resultados/snd_out.wav", snd_sample, snd_write_int)
def main():
    #quest1("lena256.png",100)
    #quest2("MasEstamosAiPraMais.wav",0.5,12520,3)   


if __name__== "__main__" :  
  main()

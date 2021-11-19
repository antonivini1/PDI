from PIL import Image
import numpy as np
import math

def which_c(number):
    if(number == 0):
        return math.sqrt(0.5)
    else:
        return 1
def dct(x,N):
    X = np.zeros(N)
    for k in range(N):
        c = which_c(k)
        summ = 0
        for n in range(N):
            summ += x[n] * math.cos(((2*n+1)*math.pi*k)/(2*N))
        X[k] = math.sqrt(2/N) * c * summ
    return X

img_src = Image.open("lena256.png")
img_arr = np.asarray(img_src)
R,C = img_arr.shape
X = np.zeros((R,C))
#img_dct = np.zeros((R,C))

# for k in range(R):
#     c_k = which_c(k)
#     for l in range(C):
#         c_l = which_c(l)
#         summ = 0
#         for m in range(R):
#             for n in range(C):
#                 summ == img_arr[m,n] * math.cos(((2*m + 1)*k*math.pi)/(2*R)) * math.cos(((2*n + 1)*l*math.pi)/(2*C))
#         X[k,l] = (2/math.sqrt(R*C)) * c_k * c_l * summ

for k in range(R):
    X[k,:] = dct(img_arr[k,:],R)
for l in range(C):
    X[:,l] = dct(X[:,l],C)

#X /= X[0][0]
img_dct = np.log(np.abs(X)+1)
img_dct *= (255.0/img_dct.max())
img_dct = Image.fromarray(img_dct)
img_dct = img_dct.convert("P")
img_dct.show()

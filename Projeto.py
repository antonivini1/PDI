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
    c_k = which_c(k)
    sum_k = 0
    for m in range(R):
        sum_k += img_arr[m,:] * math.cos(((2*m + 1)*k*math.pi)/(2*R))
    X[k,:] = math.sqrt(2/R) * c_k * sum_k
for l in range(C):
    c_l = which_c(l)
    sum_l = 0
    for n in range(C):
        sum_l += X[:,n] * math.cos(((2*n + 1)*l*math.pi)/(2*C))
    X[:,l] = math.sqrt(2/C) * c_l * sum_l

#X /= X[0][0]
img_dct = np.log(np.abs(X)+1)
img_dct *= (255.0/img_dct.max())
img_dct = Image.fromarray(img_dct)
img_dct.show()
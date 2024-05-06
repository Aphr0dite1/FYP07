import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import imageio
import scipy, scipy.misc, scipy.signal
import cv2
import sys
from PIL import Image

def computeTextureWeights(fin, sigma, sharpness):
    """
    This function computes texture weights for an input image fin using the exposure fusion framework. 
    It calculates horizontal and vertical gradients (dt0_h, dt0_v) and convolves them with Gaussian kernels (gauker_h, gauker_v). 
    Then it computes texture weights W_h and W_v based on the gradients and a sharpness parameter.
    
        - 'fin': Input image.
        - 'sigma': Parameter controlling the scale of the Gaussian kernel used for convolution.
        - 'sharpness': Parameter controlling the sharpness of the weights.
        - It calculates horizontal and vertical gradients of the input image using finite differences.
        - Gaussian kernels are convolved with these gradients.
        - Texture weights are computed based on the convolved gradients and sharpness parameter.
    
    """
    dt0_v = np.vstack((np.diff(fin, n=1, axis=0), fin[0,:]-fin[-1,:]))
    dt0_h = np.vstack((np.diff(fin, n=1, axis=1).conj().T, fin[:,0].conj().T-fin[:,-1].conj().T)).conj().T

    gauker_h = scipy.signal.convolve2d(dt0_h, np.ones((1,sigma)), mode='same')
    gauker_v = scipy.signal.convolve2d(dt0_v, np.ones((sigma,1)), mode='same')

    W_h = 1/(np.abs(gauker_h)*np.abs(dt0_h)+sharpness)
    W_v = 1/(np.abs(gauker_v)*np.abs(dt0_v)+sharpness)

    return  W_h, W_v
    
def solveLinearEquation(IN, wx, wy, lamda):
    """
    This function solves a linear equation using the given texture weights wx, wy, and a regularization parameter lambda. 
    It constructs sparse matrices Ax and Ay, computes a weight matrix D, and solves the equation to get the output image OUT.
    
        - This function solves a linear equation to achieve image smoothing.
        - 'IN': Input image.
        - 'wx', 'wy': Texture weights for horizontal and vertical directions.
        - 'lamda': Regularization parameter.
    It constructs a sparse matrix representing the linear system based on input parameters.
    Solves the linear equation using sparse matrix techniques.
    Returns the smoothed image.
        
    """
    [r, c] = IN.shape
    k = r * c
    dx =  -lamda * wx.flatten('F')
    dy =  -lamda * wy.flatten('F')
    tempx = np.roll(wx, 1, axis=1)
    tempy = np.roll(wy, 1, axis=0)
    dxa = -lamda *tempx.flatten('F')
    dya = -lamda *tempy.flatten('F')
    tmp = wx[:,-1]
    tempx = np.concatenate((tmp[:,None], np.zeros((r,c-1))), axis=1)
    tmp = wy[-1,:]
    tempy = np.concatenate((tmp[None,:], np.zeros((r-1,c))), axis=0)
    dxd1 = -lamda * tempx.flatten('F')
    dyd1 = -lamda * tempy.flatten('F')
    
    wx[:,-1] = 0
    wy[-1,:] = 0
    dxd2 = -lamda * wx.flatten('F')
    dyd2 = -lamda * wy.flatten('F')
    
    Ax = scipy.sparse.spdiags(np.concatenate((dxd1[:,None], dxd2[:,None]), axis=1).T, np.array([-k+r,-r]), k, k)
    Ay = scipy.sparse.spdiags(np.concatenate((dyd1[None,:], dyd2[None,:]), axis=0), np.array([-r+1,-1]), k, k)
    D = 1 - ( dx + dy + dxa + dya)
    A = ((Ax+Ay) + (Ax+Ay).conj().T + scipy.sparse.spdiags(D, 0, k, k)).T
    
    tin = IN[:,:]
    tout = scipy.sparse.linalg.spsolve(A, tin.flatten('F'))
    OUT = np.reshape(tout, (r, c), order='F')
    
    return OUT
    

def tsmooth(img, lamda=0.01, sigma=3.0, sharpness=0.001):
    """
    This function applies texture smoothing to the input image img using the computeTextureWeights function and then 
    solves the linear equation using solveLinearEquation function to obtain the smoothed image S.
    
        - This function applies texture smoothing to an input image.
        - 'img': Input image.
        - 'lamda', 'sigma', 'sharpness': Parameters for texture smoothing.
        - It normalizes the input image and computes texture weights.
        - Calls 'solveLinearEquation' to obtain the smoothed image.
        
    """
    I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    x = np.copy(I)
    wx, wy = computeTextureWeights(x, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lamda)
    return S

def rgb2gm(I):
    """
    This function converts an RGB image to grayscale using the geometric mean of the RGB channels.
    
        - 'I': Input RGB image.
        - It checks if the input image is RGB and then computes the grayscale image using the geometric mean formula.
    
    """
    if (I.shape[2] == 3):
        I = cv2.normalize(I.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        I = np.abs((I[:,:,0]*I[:,:,1]*I[:,:,2]))**(1/3)

    return I

def applyK(I, k, a=-0.3293, b=1.1258):
    """
    This function applies a camera model with exposure ratio k to the input image I using parameters a and b, 
    and returns the adjusted image J.
    
        - 'I': Input image.
        - 'k': Exposure ratio.
        - 'a', 'b': Parameters for the camera model.
    
    """
    f = lambda x: np.exp((1-x**a)*b)
    beta = f(k)
    gamma = k**a
    J = (I**gamma)*beta
    return J

def entropy(X):
    """
    This function calculates the entropy of an input image.
        - 'X': Input image.
    It converts the input image to uint8 format, computes pixel counts, normalizes them, and calculates entropy.
    
    """
    tmp = X * 255
    tmp[tmp > 255] = 255
    tmp[tmp<0] = 0
    tmp = tmp.astype(np.uint8)
    _, counts = np.unique(tmp, return_counts=True)
    pk = np.asarray(counts)
    pk = 1.0*pk / np.sum(pk, axis=0)
    S = -np.sum(pk * np.log2(pk), axis=0)
    return S

def maxEntropyEnhance(I, isBad, a=-0.3293, b=1.1258):
    # Implementation of max entropy enhancement
    # ...
    # Esatimate k
    tmp = cv2.resize(I, (50,50), interpolation=cv2.INTER_AREA)
    tmp[tmp<0] = 0
    tmp = tmp.real
    Y = rgb2gm(tmp)
    
    isBad = isBad * 1
    isBad = np.array(Image.fromarray(isBad).resize((50,50), Image.BICUBIC))
    
    isBad[isBad<0.5] = 0
    isBad[isBad>=0.5] = 1
    Y = Y[isBad==1]
    
    if Y.size == 0:
       J = I
       return J
    
    # Define the objective function for optimization
    f = lambda k: -entropy(applyK(Y, k))
    # Find the exposure ratio (k) that maximizes entropy
    opt_k = scipy.optimize.fminbound(f, 1, 7)
    
    # # Apply the exposure adjustment based on the optimized k
    J = applyK(I, opt_k, a, b) - 0.01
    return J

def build_is_hist(img):
    hei = img.shape[0]
    wid = img.shape[1]
    ch = img.shape[2]
    Img = np.zeros((hei+4, wid+4, ch))
    for i in range(ch):
        Img[:,:,i] = np.pad(img[:,:,i], (2,2), 'edge')
    hsv = (matplotlib.colors.rgb_to_hsv(Img))
    hsv[:,:,0] = hsv[:,:,0] * 255
    hsv[:,:,1] = hsv[:,:,1] * 255
    hsv[hsv>255] = 255
    hsv[hsv<0] = 0
    hsv = hsv.astype(np.uint8).astype(np.float64)
    fh = np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
    fv = fh.conj().T
    
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    I = hsv[:,:,2]

    dIh = scipy.signal.convolve2d(I, np.rot90(fh, 2), mode='same')
    dIv = scipy.signal.convolve2d(I, np.rot90(fv, 2), mode='same')
    dIh[dIh==0] = 0.00001
    dIv[dIv==0] = 0.00001
    dI = np.sqrt(dIh**2+dIv**2).astype(np.uint32)
    di = dI[2:hei+2,2:wid+2]
    
    dSh = scipy.signal.convolve2d(S, np.rot90(fh, 2), mode='same')
    dSv = scipy.signal.convolve2d(S, np.rot90(fv, 2), mode='same')
    dSh[dSh==0] = 0.00001
    dSv[dSv==0] = 0.00001
    dS = np.sqrt(dSh**2+dSv**2).astype(np.uint32)
    ds = dS[2:hei+2,2:wid+2]

    
    h = H[2:hei+2,2:wid+2]
    s = S[2:hei+2,2:wid+2]
    i = I[2:hei+2,2:wid+2].astype(np.uint8)
    
    Imean = scipy.signal.convolve2d(I,np.ones((5,5))/25, mode='same')
    Smean = scipy.signal.convolve2d(S,np.ones((5,5))/25, mode='same')
    
    Rho = np.zeros((hei+4,wid+4))
    for p in range(2,hei+2):
        for q in range(2,wid+2):
            tmpi = I[p-2:p+3,q-2:q+3]
            tmps = S[p-2:p+3,q-2:q+3]
            corre = np.corrcoef(tmpi.flatten('F'),tmps.flatten('F'))
            Rho[p,q] = corre[0,1]
    
    rho = np.abs(Rho[2:hei+2,2:wid+2])
    rho[np.isnan(rho)] = 0
    rd = (rho*ds).astype(np.uint32)
    Hist_I = np.zeros((256,1))
    Hist_S = np.zeros((256,1))
    
    for n in range(0,255):
        temp = np.zeros(di.shape)
        temp[i==n] = di[i==n]
        Hist_I[n+1] = np.sum(temp.flatten('F'))
        temp = np.zeros(di.shape)
        temp[i==n] = rd[i==n]
        Hist_S[n+1] = np.sum(temp.flatten('F'))

    return Hist_I, Hist_S

def dhe(img, alpha=0.5, beta=0.2):
    
    hist_i, hist_s = build_is_hist(img)
    hist_c = alpha*hist_s + (1-alpha)*hist_i
    hist_sum = np.sum(hist_c)
    hist_cum = hist_c.cumsum(axis=0)
    
    hsv = matplotlib.colors.rgb_to_hsv(img)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    i = hsv[:,:,2].astype(np.uint8)
    
    c = hist_cum / hist_sum
    s_r = (c * 255)
    i_s = np.zeros(i.shape)
    for n in range(0,255):
        i_s[i==n] = s_r[n+1]/255.0
    i_s[i==255] = 1
    
    # Adjust the intensity of enhancement by scaling i_s with beta
    i_s = i_s * beta + (1 - beta) * i / 255.0
    
    hsi_o = np.stack((h,s,i_s), axis=2)
    result = matplotlib.colors.hsv_to_rgb(hsi_o)
    
    result = result * 255
    result[result>255] = 255
    result[result<0] = 0
    return result.astype(np.uint8)

def MDS07(img, mu=0.5, a=-0.3293, b=1.1258):
    lamda = 0.5
    sigma = 5
    I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Weight matrix estimation
    t_b = np.max(I, axis=2)
    t_our = cv2.resize(tsmooth(np.array(Image.fromarray(t_b).resize((t_b.shape[1] // 2, t_b.shape[0] // 2), Image.BICUBIC)), lamda, sigma), (t_b.shape[1], t_b.shape[0]), interpolation=cv2.INTER_AREA)
    
    # Apply camera model with k(exposure ratio)
    isBad = t_our < 0.5
    J = dhe(I)  # Apply histogram equalization instead of max entropy enhancement

    # W: Weight Matrix
    t = np.zeros((t_our.shape[0], t_our.shape[1], I.shape[2]))
    for i in range(I.shape[2]):
        t[:,:,i] = t_our
    W = t**mu

    I2 = I*W
    J2 = J*(1-W)

    result = I2 + J2
    result = result * 255
    result[result > 255] = 255
    result[result<0] = 0
    return result.astype(np.uint8)

def main():
    img_name = sys.argv[1]
    img = imageio.imread(img_name)
    result = MDS07(img)
    plt.imshow(result)
    plt.show()

if __name__ == '__main__':
    main()
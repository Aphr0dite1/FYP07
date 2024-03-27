"""
These lines import necessary libraries/modules for numerical computations (numpy), 
image processing (matplotlib, imageio, PIL, cv2), and signal processing (scipy).

"""
import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy, scipy.signal
import cv2
import sys
from PIL import Image

def computeTextureWeights(fin, sigma, sharpness):
    """
    This function computes texture weights for an input image fin using the exposure fusion framework. 
    It calculates horizontal and vertical gradients (dt0_h, dt0_v) and convolves them with Gaussian kernels (gauker_h, gauker_v). 
    Then it computes texture weights W_h and W_v based on the gradients and a sharpness parameter.
    It calculates horizontal and vertical gradients of the input image using finite differences.
    Gaussian kernels are convolved with these gradients. Texture weights are computed based on the convolved gradients and sharpness parameter.
        
    Parameters:
        - fin: Input image.
        - sigma: Parameter controlling the scale of the Gaussian kernel used for convolution.
        - sharpness: Parameter controlling the sharpness of the weights.
    
    Returns:
    W_h, W_v: Texture weights for horizontal and vertical directions.
    
    We use 'np.abs' in-place operations directly on the convolved gradients 'gauker_h' and 'gauker_v'. 
    This reduces memory overhead by avoiding unnecessary copying of arrays. 
    """
    # dt0_v = np.vstack((np.diff(fin, n=1, axis=0), fin[0,:]-fin[-1,:]))
    # dt0_h = np.vstack((np.diff(fin, n=1, axis=1).conj().T, fin[:,0].conj().T-fin[:,-1].conj().T)).conj().T

    # gauker_h = scipy.signal.convolve2d(dt0_h, np.ones((1,sigma)), mode='same')
    # gauker_v = scipy.signal.convolve2d(dt0_v, np.ones((sigma,1)), mode='same')

    # W_h = 1/(np.abs(gauker_h)*np.abs(dt0_h)+sharpness)
    # W_v = 1/(np.abs(gauker_v)*np.abs(dt0_v)+sharpness)

    # return  W_h, W_v
    
    try:
        dt0_v = np.vstack((np.diff(fin, n=1, axis=0), fin[0,:] - fin[-1,:]))
        dt0_h = np.vstack((np.diff(fin, n=1, axis=1).conj().T, fin[:,0].conj().T - fin[:,-1].conj().T)).conj().T

        gauker_h = scipy.signal.convolve2d(dt0_h, np.ones((1, sigma)), mode='same')
        gauker_v = scipy.signal.convolve2d(dt0_v, np.ones((sigma, 1)), mode='same')

        np.abs(gauker_h, out=gauker_h)
        np.abs(gauker_v, out=gauker_v)

        W_h = 1 / (gauker_h * np.abs(dt0_h) + sharpness)
        W_v = 1 / (gauker_v * np.abs(dt0_v) + sharpness)

        return W_h, W_v
    
    except Exception as e:
        print("Error in computeTextureWeights:", e)
        return None, None
    
def solveLinearEquation(IN, wx, wy, lamda):
    """
    This function solves a linear equation using the given texture weights wx, wy, and a regularization parameter lambda. 
    It constructs sparse matrices Ax and Ay, computes a weight matrix D, and solves the equation to get the output image OUT.
    
    Parameters:
        - IN: Input image.
        - wx, wy: Texture weights for horizontal and vertical directions.
        - lamda: Regularization parameter.
    
    Returns:
        OUT: Smoothed image.
    
    It constructs a sparse matrix representing the linear system based on input parameters.
    Solves the linear equation using sparse matrix techniques.
    Returns the smoothed image.
        
    """
    try:
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
    except Exception as e:
        print("Error in solveLinearEquation:", e)
        return None
   

def tsmooth(img, lamda=0.01, sigma=3.0, sharpness=0.001):
    """
    This function applies texture smoothing to the input image img using the computeTextureWeights function and then 
    solves the linear equation using solveLinearEquation function to obtain the smoothed image S.
    
    Parameters:
        - img: Input image.
        - lamda, sigma, sharpness: Parameters for texture smoothing.
    
    Returns:
        S: Smoothed image.
    
    It normalizes the input image and computes texture weights.
    Calls 'solveLinearEquation' to obtain the smoothed image.
        
    """
    try:
        I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        x = np.copy(I)
        wx, wy = computeTextureWeights(x, sigma, sharpness)
        S = solveLinearEquation(I, wx, wy, lamda)
        return S
    
    except Exception as e:
        print("Error in tsmooth:", e)
        return None

def rgb2gm(I):
    """
    This function converts an RGB image to grayscale using the geometric mean of the RGB channels.
    
    Parameters:
        - I: Input RGB image.
    
    Returns:
        Grayscale image.
    
    It checks if the input image is RGB and then computes the grayscale image using the geometric mean formula.
    
    """
    try:
        if (I.shape[2] == 3):
            I = cv2.normalize(I.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            I = np.abs((I[:,:,0]*I[:,:,1]*I[:,:,2]))**(1/3)

        return I
    
    except Exception as e:
        print("Error in rgb2gm:", e)
        return None

def applyK(I, k, a=-0.3293, b=1.1258):
    """
    This function applies a camera model with exposure ratio k to the input image I using parameters a and b, 
    and returns the adjusted image J.
    
    Parameters:
        - 'I': Input image.
        - 'k': Exposure ratio.
        - 'a', 'b': Parameters for the camera model.
    
    Returns:
        Adjusted image.
    
    """
    try:
        f = lambda x: np.exp((1-x**a)*b)
        beta = f(k)
        gamma = k**a
        J = (I**gamma)*beta
        return J
    
    except Exception as e:
        print("Error in applyK:", e)
        return None

def entropy(X):
    """
    This function calculates the entropy of an input image.
    Entropy quantifies the amount of information or the level of complexity within the image. 
    A higher entropy value indicates greater unpredictability or disorder in pixel values, while a lower entropy value suggests more regularity or predictability.
        
        - In this case, entropy is used as a measure of image texture or complexity.
    
    In simpler terms, entropy measures how much information is needed to represent the distribution of pixel values in the image. 
    A higher entropy value implies a broader range of pixel intensities and more complexity in the image, while a lower entropy value indicates less complexity and a more uniform distribution of pixel values.
    
    Parameters:
        - X: Input image.
    
    Returns:
        Entropy value.
        
    It converts the input image to uint8 format, computes pixel counts, normalizes them, and calculates entropy.
    
    """
    # tmp = X * 255
    # tmp[tmp > 255] = 255
    # tmp[tmp<0] = 0
    # tmp = tmp.astype(np.uint8)
    # _, counts = np.unique(tmp, return_counts=True)
    # pk = np.asarray(counts)
    # pk = 1.0*pk / np.sum(pk, axis=0)
    # S = -np.sum(pk * np.log2(pk), axis=0)
    # return S
    
    try:
        # Compute pixel counts
        counts, _ = np.histogram(X, bins=256, range=(0, 1))

        # Normalize counts
        pk = counts / np.sum(counts)

        # Calculate entropy
        S = -np.sum(pk * np.log2(pk + 1e-10))  # Adding a small value to prevent log(0)

        return S
     
    except Exception as e:
        print("Error in entrophy:", e)
        return None

def maxEntropyEnhance(I, isBad, a=-0.3293, b=1.1258):
    """
    This function enhances an image based on maximum entropy estimation.
    
    The goal is to adjust the exposure of the input image to maximize its entropy, thereby increasing its information content and visual quality.
    
    Parameters:
        - I: Input image.
        - isBad: Indicator of bad pixels.
        - a, b: Parameters for the camera model.
    
    Returns:
        Enhanced image.
        
    It estimates the exposure ratio 'k' based on maximum entropy estimation.
    It utilizes a downscaled version of the input image to reduce computational complexity.
    Pixels are classified as "bad" or "good" based on their intensity values.
    Applies the camera model to enhance the input image and returns the enhanced image
    
    """
    # # Esatimate k
    # tmp = cv2.resize(I, (50,50), interpolation=cv2.INTER_AREA)
    # tmp[tmp<0] = 0
    # tmp = tmp.real
    # Y = rgb2gm(tmp)
    
    # isBad = isBad * 1
    # isBad = np.array(Image.fromarray(isBad).resize((50,50), Image.BICUBIC))
    
    # isBad[isBad<0.5] = 0
    # isBad[isBad>=0.5] = 1
    # Y = Y[isBad==1]
    
    # if Y.size == 0:
    #    J = I
    #    return J
    
    # # Define the objective function for optimization
    # f = lambda k: -entropy(applyK(Y, k))
    # # Find the exposure ratio (k) that maximizes entropy
    # opt_k = scipy.optimize.fminbound(f, 1, 7)
    
    # # # Apply the exposure adjustment based on the optimized k
    # J = applyK(I, opt_k, a, b) - 0.01
    # return J
    
    try:
        def entropy_neg(k):
            """
            Objective function to minimize: negative entropy.
            
            Parameters:
                - k: Exposure ratio.

            Returns:
                Negative entropy value.
                
            This function calculates the negative entropy of the image after applying the exposure adjustment
            with the given exposure ratio 'k'. It downscales the input image and selects valid pixels based on
            the 'isBad' indicator. Then it computes the grayscale image and calculates the negative entropy.
            The goal is to find the exposure ratio 'k' that maximizes the entropy of the image, enhancing its
            information content and visual quality.
            """
            tmp = cv2.resize(I, (50, 50), interpolation=cv2.INTER_AREA)
            tmp[tmp < 0] = 0
            tmp = tmp.real
            Y = rgb2gm(tmp)

            isBadTmp = isBad * 1
            isBadTmp = np.array(Image.fromarray(isBadTmp).resize((50, 50), Image.BICUBIC))

            isBadTmp[isBadTmp < 0.5] = 0
            isBadTmp[isBadTmp >= 0.5] = 1
            Y = Y[isBadTmp == 1]

            if Y.size == 0:
                return np.inf  # Return positive infinity if no valid pixels

            return -entropy(applyK(Y, k))

        # Find the exposure ratio (k) that maximizes entropy
        opt_res = scipy.optimize.minimize_scalar(entropy_neg, bounds=(1, 7), method='bounded')

        # Apply the exposure adjustment based on the optimized k
        J = applyK(I, opt_res.x, a, b) - 0.01
        return J
    
    except Exception as e:
        print("Error in maxEntrophyEnhance:", e)
        return None
        
    

def mds07_fusion(img, mu=0.5, a=-0.3293, b=1.1258):
    """
    This function, MDS07, performs the main processing steps:
        - It initializes parameters such as lambda and sigma.
        - It normalizes the input image img.
        - It estimates a weight matrix (t_our) based on the maximum intensity of the input image (t_b) and applies texture smoothing (tsmooth) to it.
        - It identifies "bad" pixels in the smoothed image and enhances the input image using the maxEntropyEnhance function.
        - It constructs a weight matrix W based on the smoothed image.
        - It fuses the input image I and enhanced image J based on the constructed weights.
        - It adjusts the intensity range of the fused image and returns the result.
    
    Parameters:
        - img: Input image.
        - mu, a, b: Parameters for the exposure fusion framework.
    
    Returns:
        Processed image.    
    """
    try:
        lamda = 0.5
        sigma = 5
        I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        # Weight matrix estimation
        t_b = np.max(I, axis=2)
        t_our = cv2.resize(tsmooth(np.array(Image.fromarray(t_b).resize((t_b.shape[1] // 2, t_b.shape[0] // 2), Image.BICUBIC)), lamda, sigma), (t_b.shape[1], t_b.shape[0]), interpolation=cv2.INTER_AREA)
        
        # Apply camera model with k(exposure ratio)
        isBad = t_our < 0.5
        J = maxEntropyEnhance(I, isBad)

        # W: Weight Matrix
        t = np.zeros((t_our.shape[0], t_our.shape[1], I.shape[2]))
        for i in range(I.shape[2]):
            t[:,:,i] = t_our
        W = t**mu

        # Apply fusion based on weights
        I2 = I*W
        J2 = J*(1-W)

        result = I2 + J2
        result = result * 255
        result[result > 255] = 255
        result[result<0] = 0
        return result.astype(np.uint8)
    
    except Exception as e:
        print("Error in mds07_fusion:", e)
        return None

def main():
    try:
        if len(sys.argv) < 2:
            raise ValueError("Please provide the image file name as an argument.")
        
        img_name = sys.argv[1]
        
        try:
            img = imageio.imread(img_name)
        except FileNotFoundError:
            print("Error: File not found.")
            return
        except Exception as e:
            print("Error:", e)
            return 

        try:
            result = mds07_fusion(img)
        except Exception as e:
            print("Error:", e)
            return
        
        if result is not None:
            plt.imshow(result)
            plt.show()
        
        else:
            print("Error occurred during image processing.")
            
    except Exception as e:
        print("Error in main:", e)

if __name__ == '__main__':
    main()

from torch import randperm, utils
from torch._utils import _accumulate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from PIL import Image

class Subset(utils.data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)



def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]

# Quizas deberia eliminar 3d o limpiar
def plot_embedding(X, merged, title = None, classes=11., showimage=True, distPl=0.006, onlyRoman=False):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    
    plt.figure()
    ax = plt.subplot(111)
    ax.set_facecolor('xkcd:white')
    """
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(merged.iloc[i][1]),
                 color=plt.cm.Set1(int(merged.iloc[i][1]) / float(classes)),
                 fontdict={'weight': 'bold', 'size': 9})
    """
    """
    for i in range(X.shape[0]):
        if int(merged.iloc[i][1]) == 22:
            plt.plot([X[i, 0]], [X[i, 1]], 'X', c="black", markersize=10)
            plt.plot([X[i, 0]], [X[i, 1]], 'X', c='black', markersize=8)
        else:
            plt.plot([X[i, 0]], [X[i, 1]], 'o', c="black", markersize=6)
            plt.plot([X[i, 0]], [X[i, 1]], 'o',c=plt.cm.Set3(int(merged.iloc[i][1])), markersize=4)
    """
  
    
    if showimage and hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])
        for i in range(merged.shape[0]):           
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            
            
            if np.min(dist) < distPl: #6e-4:
                # don't show points that are too close
                continue

            shown_images = np.r_[shown_images, [X[i]]]
            image =  Image.open(merged.iloc[i][0])
            inverted_image = image #PIL.ImageOps.invert(image)
            inverted_image.thumbnail((40, 40), Image.ANTIALIAS)
            
            props = dict(facecolor=plt.cm.Set3(int(merged.iloc[i][1])), alpha=1, lw=1)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(inverted_image, cmap=plt.cm.gray),
                X[i]+0.030, bboxprops=props)
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    #cbar = plt.colorbar()
    if title is not None:
        plt.title(title)


import numpy as np
import cv2
from skimage import measure
def landmarks(img_grey, N = 50):
    thresh = 200
    ret,img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    if img.ndim == 2:
        img_s = np.ones((img.shape[0] + 100, img.shape[0] + 100)) * 255
        img_s[50:-50, 50:-50] = img
        img = img_s
        contours = measure.find_contours(img, 0.5)
        #fig = plt.figure(figsize=(7, 7))
        #ax = fig.add_subplot(111)
        #ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)

        # for n, contour in enumerate(contours):
        #print(contours[0].shape)
        contour = contours[0]
        #ax.plot(contour[:, 1], contour[:, 0], linewidth=5)
        # resample_contour = contour[np.random.choice(contour.shape[0], 150, replace=False), :]
        resample_contour = interpcurve(N,  contour[:, 0],  contour[:, 1])
        # print(resample_contour[:4, 0], resample_contour[:4, 1], resample_contour[:4].ravel())
        #df_semilandmarks.loc[index] = [id_name, classe_name] + list(resample_contour.ravel())
        #ax.plot(resample_contour[:, 1], resample_contour[:, 0], 'om', linewidth=5)
        #plt.savefig('output/landmarked_'+id_name)
        #plt.show()
        return resample_contour
    
def interpcurve(N, pX, pY):
    #equally spaced in arclength
    N = np.transpose(np.linspace(0, 1, N))
    #how many points will be uniformly interpolated?
    nt = N.size

    #number of points on the curve
    n = pX.size
    pxy = np.array((pX, pY)).T
    p1 = pxy[0,:]
    pend = pxy[-1,:]
    last_segment = np.linalg.norm(np.subtract(p1, pend))
    epsilon= 10 * np.finfo(float).eps

    #IF the two end points are not close enough lets close the curve
    if last_segment > epsilon * np.linalg.norm(np.amax(abs(pxy), axis=0)):
        pxy = np.vstack((pxy, p1))
        nt = nt + 1

    pt = np.zeros((nt, 2))

    #Compute the chordal arclength of each segment.
    chordlen = (np.sum(np.diff(pxy, axis=0) ** 2, axis=1)) ** (1 / 2)
    #Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    #cumulative arclength
    cumarc = np.append(0, np.cumsum(chordlen))

    tbins= np.digitize(N, cumarc) # bin index in which each N is in

    #catch any problems at the ends
    tbins[np.where(tbins<=0 | (N<=0))]=1
    tbins[np.where(tbins >= n | (N >= 1))] = n - 1      

    s = np.divide((N - cumarc[tbins]), chordlen[tbins-1])
    pt = pxy[tbins,:] + np.multiply((pxy[tbins,:] - pxy[tbins-1,:]), (np.vstack([s]*2)).T)

    return pt 



def segmentation(img, vertical):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(img, cmap=plt.cm.gray)
    if vertical:
        background_0 = np.ones((img.shape)) + 254
        background_0[img.shape[0]//2:, :] = img[img.shape[0]//2:,:] 
        #plt.imshow(background_0, cmap=plt.cm.gray)
        #fig = plt.figure()

        background_1 = np.ones((img.shape)) + 254
        background_1[:img.shape[0]//2, :] = img[:img.shape[0]//2,:] 
        #plt.imshow(background_1, cmap=plt.cm.gray)
        #fig = plt.figure()
         

    else:
        background_1 = np.ones((img.shape)) + 254
        background_1[:, :img.shape[0]//2] = img[:,:img.shape[0]//2] 
        #plt.imshow(background_0, cmap=plt.cm.gray)
        #fig = plt.figure()


        background_0 = np.ones((img.shape)) + 254
        background_0[:, img.shape[0]//2:] = img[:,img.shape[0]//2:] 
        #plt.imshow(background_1, cmap=plt.cm.gray)
        #fig = plt.figure()
    
    return background_0, background_1

def plotTwoImages(img_1, img_2, title):
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)

    plt.imshow(img_1)
    plt.title(title[0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    
    plt.imshow(img_2)
    plt.xticks([])
    plt.yticks([])
    plt.title(title[1])
    
def plotLandmarks(landmarks_img_1_part_2, landmarks_img_2_part_2, title, comparateShow=True):
    plt.figure(figsize=(12,5))
    size = (1, 2)
    if comparateShow:
        size = (1, 3)
    plt.subplot( size[0],  size[1], 1)
    plt.plot(landmarks_img_1_part_2[:,0], landmarks_img_1_part_2[:,1], '-o')
    plt.fill(landmarks_img_1_part_2[:, 0] , landmarks_img_1_part_2[:, 1], 'k')
    plt.title(title[0])
    plt.xticks([])
    plt.yticks([])


    plt.subplot(size[0], size[1], 2)
    plt.plot(landmarks_img_2_part_2[:,0], landmarks_img_2_part_2[:,1], '-o')
    plt.fill(landmarks_img_2_part_2[:, 0] , landmarks_img_2_part_2[:, 1], 'k')
    plt.title(title[1])
    plt.xticks([])
    plt.yticks([])


    if comparateShow:
        plt.subplot(size[0], size[1], 3)
        plt.plot(landmarks_img_1_part_2[:,0], landmarks_img_1_part_2[:,1], '-o')
        plt.plot(landmarks_img_2_part_2[:,0], landmarks_img_2_part_2[:,1], '-o')
        plt.title(title[2])
        plt.xticks([])
        plt.yticks([])

def plotLandmarks2(landmarks_img_1_part_2, landmarks_img_2_part_2, title, comparateShow=True):
    plt.figure(figsize=(12,5))
    fig, (ax1) = plt.subplots(1, 1, figsize=(12,5))
    ax1.axis("off")
    landmarks_img_1_part_2 = landmarks_img_1_part_2 * -1
    landmarks_img_2_part_2 = landmarks_img_2_part_2 * -1
    ax1.plot(landmarks_img_1_part_2[:,1], landmarks_img_1_part_2[:,0], '-o')
    ax1.plot(landmarks_img_2_part_2[:,1], landmarks_img_2_part_2[:,0], '-o')
    plt.title(title[2])
    ax1.set_aspect(2)
    plt.xticks([])
    plt.yticks([])
    

def plotLandmarksItem(fig, ax1, landmarks_img_1_part_2, landmarks_img_2_part_2, title, comparateShow=True):
    ax1.axis("off")
    landmarks_img_1_part_2 = landmarks_img_1_part_2 * -1
    landmarks_img_2_part_2 = landmarks_img_2_part_2 * -1
    ax1.plot(landmarks_img_1_part_2[:,1], landmarks_img_1_part_2[:,0], '-o')
    ax1.plot(landmarks_img_2_part_2[:,1], landmarks_img_2_part_2[:,0], '-o')
    ax1.set_title(title)
    ax1.set_aspect(1)
    

def plotLandmarksALL(img_0, img_1, landmarks_img_1_part_1, landmarks_img_2_part_1, landmarks_img_1_part_2, landmarks_img_2_part_2, title):
    plt.figure(figsize=(12,5))
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(12,5))
    ax0.axis("off")
    ax0.set_title('INPUT')
    ax3.axis("off")
    ax3.set_title('OUTPUT')
    plt.xticks([])
    plt.yticks([])
    ax0.imshow(img_0)
    plotLandmarksItem(fig, ax1, landmarks_img_1_part_1, landmarks_img_2_part_1, title[0])
    plotLandmarksItem(fig, ax2, landmarks_img_1_part_2, landmarks_img_2_part_2, title[1])
    ax3.imshow(img_1)
    plt.xticks([])
    plt.yticks([])




if __name__ == "__main__":
    pass
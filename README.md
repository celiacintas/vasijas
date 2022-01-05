# 2D and 3D pottery geometric analysis and generation
In this repo you can find:

- Preprocessing pipeline with skimage to extract contourns and semilandmarks. solid revolution based on semilandmarks.
- Dimension reduction over semilandmarks and over raw pixel space and clustering.
- CNN classification over the 11 pottery classes.
- 2D GAN trained network over pixel space to generate new pottery contourns.
- 3D GAN trained network over voxel space to generate new 3D pottery.

## 2D GAN generation on 100 epochs
![2D GAN](imagenes/GAN_epochs.gif)

## 3D GAN based on stl models of solid revolution of 100 semilandmarks
![3D GAN](imagenes/gan3d_100epochs.png)


### Semilandmarks examples
![semilandmarks](imagenes/vasija_contorno_1_semilandmark.png)
![semilandmarks](imagenes/vasija_contorno_2_semilandmark.png)


### TSNE over raw pixel space
![tsne](imagenes/tsne.png)
![tsne](imagenes/tsne_2.png)

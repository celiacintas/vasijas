# Automatic feature extraction and classification of Iberian ceramics based on deep convolutional networks
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.culher.2019.06.005-f9f107.svg)](https://doi.org/10.1016/j.culher.2019.06.005)

## Abstract

Accurate classification of pottery vessels is a key aspect in several archaeological inquiries, including documentation of changes in style and ornaments, inference of chronological and ethnic groups, trading routes analyses, and many other matters. We present an unsupervised method for automatic feature extraction and classification of wheel-made vessels. A convolutional neural network was trained with a profile image database from Iberian wheel made pottery vessels found in the upper valley of the Guadalquivir River (Spain). During the design of the model, data augmentation and regularization techniques were implemented to obtain better generalization outcomes. The resulting model is able to provide classification on profile images automatically, with an accuracy mean score of 0.9013. Such computation methods will enhance and complement research on characterization and classification of pottery assemblages based on fragments.

## Dataset

The raw information belong to binary profile images, corresponding to Iberian wheel made pottery from various archaeological sites of the upper valley of the Guadalquivir River (Spain). Reference classification has been done by an expert group, based on morphological criteria, taking into account the presence or absence of certain parts, such as lip, neck, body, base and handles, and the ratios between their corresponding sizes. According to these criteria, vessels can be classified as belonging to one of 11 different classes, each one with different number of elements. Nine of them correspond to closed shapes, and the two remaining correspond to open shapes. The available images consist of a profile view of the pottery, where image resolutions (in pixels), corresponding to size scale, may vary according to the acquisition settings.



## TSNE over raw pixel space
![tsne](imagenes/tsne.png)
![tsne](imagenes/tsne_2.png)


## Citation

```Latex
@article{CINTAS2020106,
title = {Automatic feature extraction and classification of Iberian ceramics based on deep convolutional networks},
journal = {Journal of Cultural Heritage},
volume = {41},
pages = {106-112},
year = {2020},
issn = {1296-2074},
doi = {https://doi.org/10.1016/j.culher.2019.06.005},
url = {https://www.sciencedirect.com/science/article/pii/S1296207418307775},
author = {Celia Cintas and Manuel Lucena and José Manuel Fuertes and Claudio Delrieux and Pablo Navarro and Rolando González-José and Manuel Molinos},
keywords = {Deep learning, Convolutional networks, Pottery profiles, Typologies},
}
```

## More Papers

- Reconstruction of Iberian ceramic potteries using auto-encoder generative adversarial networks

[![DOI](https://img.shields.io/badge/DOI-10.1016/j.culher.2021.01.003-f9f107.svg)](https://doi.org/10.1016/j.culher.2021.01.003)
[![CODE](https://img.shields.io/badge/CODE-73ff9.svg)](https://github.com/celiacintas/vasijas/tree/unsupervised)

- Reconstruction of Iberian ceramic potteries using auto-encoder generative adversarial networks

[![CODE](https://img.shields.io/badge/CODE-73ff9.svg)](https://github.com/celiacintas/vasijas/tree/iberianGAN)

## Other expermitens in this repo

- Preprocessing pipeline with skimage to extract contourns and semilandmarks. solid revolution based on semilandmarks.
- Dimension reduction over semilandmarks and over raw pixel space and clustering.
- 2D GAN trained network over pixel space to generate new pottery contourns.
- 3D GAN trained network over voxel space to generate new 3D pottery.

## 2D GAN generation on 100 epochs
![2D GAN](imagenes/GAN_epochs.gif)

## 3D GAN based on stl models of solid revolution of 100 semilandmarks
![3D GAN](imagenes/gan3d_100epochs.png)

### Semilandmarks examples
![semilandmarks](imagenes/vasija_contorno_1_semilandmark.png)
![semilandmarks](imagenes/vasija_contorno_2_semilandmark.png)


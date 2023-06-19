# Learning feature representation of Iberian ceramics with automatic classification models

[![DOI](https://img.shields.io/badge/DOI-10.1016/j.culher.2021.01.003-f9f107.svg)](https://doi.org/10.1016/j.culher.2021.01.003)

## Abstract

In Cultural Heritage inquiries, a common requirement is to establish time-based trends between archaeological artifacts belonging to different periods of a given culture, enabling among other things to determine chronological inferences with higher accuracy and precision.
Among these, pottery vessels are significantly useful, given their relative abundance in most archaeological sites.
However, this very abundance makes difficult and complex an accurate representation, since no two of these vessels are identical, and therefore classification criteria must be justified and applied.
For this purpose, we propose the use of deep learning architectures to extract automatically learned features without prior knowledge or engineered features.
By means of transfer learning, we retrained a Residual Neural Network with a binary image database of Iberian wheel-made pottery vessels' profiles. 
These vessels pertain to archaeological sites located in the upper valley of the Guadalquivir River (Spain).
The resulting model can provide an accurate feature representation space, which can automatically classify profile images, achieving a mean accuracy of $0.96$ with an $f$-measure of $0.96$. 
This accuracy is remarkably higher than other state-of-the-art machine learning approaches, where several feature extraction techniques were applied together with multiple classifier models.
These results provide novel strategies to current research in automatic feature representation and classification of different objects of study within the Archaeology domain. 

## Dataset

The profile images correspond to domain experts' drawings from Iberian wheel-made vessels collected in various archaeological sites located along the upper valley of the Guadalquivir River (Spain). The classification of the vessels corresponds to eleven different classes based on the shape. These include the forms of the lip, neck, body, base, and handles, and the relative ratios between their sizes.
Nine of these classes correspond to closed vessel shapes, while the two other belong to open ones.

## UMAP visualization for the Pretraind ResNet with feature extraction
![](result_plot/resNet/unap_resnet.png)

## UMAP visualization for the CNN with feature extraction
![](result_plot/custom_cnn/unap_custom_cnn.png)

## Normalized confusion Matrix of the predicted results of the SVC
![](result_plot/resNet/svc_confusion_matrix_resnet.png)

## Citation

```Latex
@article{NAVARRO202165,
title = {Learning feature representation of Iberian ceramics with automatic classification models},
journal = {Journal of Cultural Heritage},
volume = {48},
pages = {65-73},
year = {2021},
issn = {1296-2074},
doi = {https://doi.org/10.1016/j.culher.2021.01.003},
url = {https://www.sciencedirect.com/science/article/pii/S1296207421000042},
author = {Pablo Navarro and Celia Cintas and Manuel Lucena and José Manuel Fuertes and Claudio Delrieux and Manuel Molinos},
keywords = {Representation learning, Iberian pottery, Deep learning},
}
```

## More Papers

- IberianVoxel: Automatic Completion of Iberian Ceramics for Cultural Heritage Studies
  
[![CODE](https://camo.githubusercontent.com/cab0ba8ebc53130e4e17ecf07c91c58c3d369da13fd2b4dabfb495be044a5c6c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f434f44452d37336666392e737667)](https://github.com/celiacintas/vasijas/tree/iberianVox)

- Reconstruction of Iberian ceramic potteries using auto-encoder generative adversarial networks
  
[![DOI](https://camo.githubusercontent.com/9cd5cadde3971729b0c553a8df0c851ea4ba193d5a25b30bfd5ec91a6e849f8d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f444f492d31302e313033382f7334313539382e3032322e31343931302e372d6639663130372e737667)](https://doi.org/10.1038/s41598-022-14910-7)
[![CODE](https://camo.githubusercontent.com/cab0ba8ebc53130e4e17ecf07c91c58c3d369da13fd2b4dabfb495be044a5c6c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f434f44452d37336666392e737667)](https://github.com/celiacintas/vasijas/tree/iberianGAN)

- Automatic feature extraction and classification of Iberian ceramics based on deep convolutional networks

[![DOI](https://camo.githubusercontent.com/8139bfdbfa153ff4989dac3f4622ece7adff84137be5916b26d300acdaf06aed/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f444f492d31302e313031362f6a2e63756c6865722e323031392e30362e3030352d6639663130372e737667)](https://doi.org/10.1016/j.culher.2019.06.005)
[![CODE](https://camo.githubusercontent.com/cab0ba8ebc53130e4e17ecf07c91c58c3d369da13fd2b4dabfb495be044a5c6c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f434f44452d37336666392e737667)](https://github.com/celiacintas/vasijas/tree/classification)



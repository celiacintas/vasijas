# Papers in this repository

## IberianVoxel: Automatic Completion of Iberian Ceramics for Cultural Heritage Studies

Accurate completion of archaeological artifacts is a critical aspect in several archaeological studies, including documentation of variations in style, inference of chronological and ethnic groups, and trading routes trends, among many others.
However, most available pottery is fragmented, leading to missing textural and morphological cues.
Currently, the reassembly and completion of fragmented ceramics is a daunting and time-consuming task, done almost exclusively by hand, which requires the physical manipulation of the fragments.
To overcome the challenges of manual reconstruction, reduce the materials' exposure and deterioration, and improve the quality of reconstructed samples, we present IberianVoxel, a novel 3D Autoencoder Generative Adversarial Network (3D AE-GAN) framework tested on an extensive database with complete and fragmented references.
We generated a collection of $1001$ 3D voxelized samples and their fragmented references from Iberian wheel-made pottery profiles.
The fragments generated are stratified into different size groups and across multiple pottery classes.
Lastly, we provide quantitative and qualitative assessments to measure the quality of the reconstructed voxelized samples by our proposed method and archaeologists' evaluation.

<!-- [![DOI](https://img.shields.io/badge/DOI-10.1038/s41598.022.14910.7-f9f107.svg)](https://doi.org/10.1038/s41598-022-14910-7) -->
[![DOI](https://img.shields.io/badge/DOI-10.24963/ijcai.2023-f9f107.svg)](https://doi.org/10.24963/ijcai.2023/647)
[![CODE](https://camo.githubusercontent.com/cab0ba8ebc53130e4e17ecf07c91c58c3d369da13fd2b4dabfb495be044a5c6c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f434f44452d37336666392e737667)](https://github.com/celiacintas/vasijas/tree/iberianVox)
[![DEMO](https://img.shields.io/badge/DEMO-73ff9.svg)](https://huggingface.co/spaces/pablo1n7/iberianVoxels)

### Citation

```

@inproceedings{ijcai2023p647,
  title     = {IberianVoxel: Automatic Completion of Iberian Ceramics for Cultural Heritage Studies},
  author    = {Navarro, Pablo and Cintas, Celia and Lucena, Manuel and Fuertes, José Manuel and Rueda, Antonio and Segura, Rafael and Ogayar-Anguita, Carlos and González-José, Rolando and Delrieux, Claudio},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {5833--5841},
  year      = {2023},
  month     = {8},
  note      = {AI and Arts},
  doi       = {10.24963/ijcai.2023/647},
  url       = {https://doi.org/10.24963/ijcai.2023/647},
}

```

## Reconstruction of Iberian ceramic potteries using auto-encoder generative adversarial networks

Several aspects of past culture, including historical trends, are inferred from time-based patterns observed in archaeological artifacts belonging to different periods. Because its presence and variation give important clues about the Neolithic revolution and given their relative abundance in most archaeological sites, ceramic potteries are significantly helpful in this purpose. Nonetheless, most available pottery is fragmented, leading to missing morphological information. Currently, the reassembly of fragmented objects from a collection of hundreds of mixed fragments is manually done by experts. To overcome the pitfalls of manual reconstruction and improve the quality of reconstructed samples, here we present a Generative Adversarial Network (GAN) framework tested on an extensive database with complete and fragmented references. Using customized GANs, we trained a model with 1072 samples corresponding to Iberian wheel-made pottery profiles belonging to archaeological sites located in the upper valley of the Guadalquivir River (Spain). We provide quantitative and qualitative assessments to measure the quality of the reconstructed samples, along with domain expert evaluation with archaeologists. The resulting framework is a possible way to facilitate pottery reconstruction from partial fragments of an original piece.

[![DOI](https://camo.githubusercontent.com/9cd5cadde3971729b0c553a8df0c851ea4ba193d5a25b30bfd5ec91a6e849f8d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f444f492d31302e313033382f7334313539382e3032322e31343931302e372d6639663130372e737667)](https://doi.org/10.1038/s41598-022-14910-7)
[![CODE](https://camo.githubusercontent.com/cab0ba8ebc53130e4e17ecf07c91c58c3d369da13fd2b4dabfb495be044a5c6c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f434f44452d37336666392e737667)](https://github.com/celiacintas/vasijas/tree/iberianGAN)
[![DEMO](https://img.shields.io/badge/DEMO-73ff9.svg)](https://huggingface.co/spaces/pablo1n7/iberianGAN)

### Citation

```
@article{navarro2022reconstruction,
  title={Reconstruction of Iberian ceramic potteries using generative adversarial networks},
  author={Navarro, Pablo and Cintas, Celia and Lucena, Manuel and Fuertes, Jos{\'e} Manuel and Segura, Rafael and Delrieux, Claudio and Gonz{\'a}lez-Jos{\'e}, Rolando},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={10644},
  year={2022},
  publisher={Nature Publishing Group UK London}
}

```

## Learning feature representation of Iberian ceramics with automatic classification models

In Cultural Heritage inquiries, a common requirement is to establish time-based trends between archaeological artifacts belonging to different periods of a given culture, enabling among other things to determine chronological inferences with higher accuracy and precision.
Among these, pottery vessels are significantly useful, given their relative abundance in most archaeological sites.
However, this very abundance makes difficult and complex an accurate representation, since no two of these vessels are identical, and therefore classification criteria must be justified and applied.
For this purpose, we propose the use of deep learning architectures to extract automatically learned features without prior knowledge or engineered features.
By means of transfer learning, we retrained a Residual Neural Network with a binary image database of Iberian wheel-made pottery vessels' profiles.
These vessels pertain to archaeological sites located in the upper valley of the Guadalquivir River (Spain).
The resulting model can provide an accurate feature representation space, which can automatically classify profile images, achieving a mean accuracy of $0.96$ with an $f$-measure of $0.96$.
This accuracy is remarkably higher than other state-of-the-art machine learning approaches, where several feature extraction techniques were applied together with multiple classifier models.
These results provide novel strategies to current research in automatic feature representation and classification of different objects of study within the Archaeology domain.

[![DOI](https://camo.githubusercontent.com/b4f5d9ec8bf9e3ac10e481b99c8f3dd8b660b0b85a9caef1c78daed60974f724/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f444f492d31302e313031362f6a2e63756c6865722e323032312e30312e3030332d6639663130372e737667)](https://doi.org/10.1016/j.culher.2021.01.003)
[![CODE](https://camo.githubusercontent.com/cab0ba8ebc53130e4e17ecf07c91c58c3d369da13fd2b4dabfb495be044a5c6c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f434f44452d37336666392e737667)](https://github.com/celiacintas/vasijas/tree/unsupervised)

### Citation

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

## Automatic feature extraction and classification of Iberian ceramics based on deep convolutional networks

Accurate classification of pottery vessels is a key aspect in several archaeological inquiries, including documentation of changes in style and ornaments, inference of chronological and ethnic groups, trading routes analyses, and many other matters. We present an unsupervised method for automatic feature extraction and classification of wheel-made vessels. A convolutional neural network was trained with a profile image database from Iberian wheel made pottery vessels found in the upper valley of the Guadalquivir River (Spain). During the design of the model, data augmentation and regularization techniques were implemented to obtain better generalization outcomes. The resulting model is able to provide classification on profile images automatically, with an accuracy mean score of 0.9013. Such computation methods will enhance and complement research on characterization and classification of pottery assemblages based on fragments.

[![DOI](https://camo.githubusercontent.com/8139bfdbfa153ff4989dac3f4622ece7adff84137be5916b26d300acdaf06aed/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f444f492d31302e313031362f6a2e63756c6865722e323031392e30362e3030352d6639663130372e737667)](https://doi.org/10.1016/j.culher.2019.06.005)
[![CODE](https://camo.githubusercontent.com/cab0ba8ebc53130e4e17ecf07c91c58c3d369da13fd2b4dabfb495be044a5c6c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f434f44452d37336666392e737667)](https://github.com/celiacintas/vasijas/tree/classification)

### Citation

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

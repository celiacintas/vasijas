# Papers in this repository

## Automatic feature extraction and classification of Iberian ceramics based on deep convolutional networks

Accurate classification of pottery vessels is a key aspect in several archaeological inquiries, including documentation of changes in style and ornaments, inference of chronological and ethnic groups, trading routes analyses, and many other matters. We present an unsupervised method for automatic feature extraction and classification of wheel-made vessels. A convolutional neural network was trained with a profile image database from Iberian wheel made pottery vessels found in the upper valley of the Guadalquivir River (Spain). During the design of the model, data augmentation and regularization techniques were implemented to obtain better generalization outcomes. The resulting model is able to provide classification on profile images automatically, with an accuracy mean score of 0.9013. Such computation methods will enhance and complement research on characterization and classification of pottery assemblages based on fragments.

[![DOI](https://img.shields.io/badge/DOI-10.1016/j.culher.2019.06.005-f9f107.svg)](https://doi.org/10.1016/j.culher.2019.06.005)
[![CODE](https://img.shields.io/badge/CODE-73ff9.svg)](https://github.com/celiacintas/vasijas/tree/classification)

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

[![DOI](https://img.shields.io/badge/DOI-10.1016/j.culher.2021.01.003-f9f107.svg)](https://doi.org/10.1016/j.culher.2021.01.003)
[![CODE](https://img.shields.io/badge/CODE-73ff9.svg)](https://github.com/celiacintas/vasijas/tree/unsupervised)

## Reconstruction of Iberian ceramic potteries using auto-encoder generative adversarial networks

Several aspects of past culture, including historical trends, are inferred from time-based patterns observed in archaeological artifacts belonging to different periods. Ceramic potteries are significantly helpful in this purpose, given their relative abundance in most archaeological sites. Nonetheless, most available pottery is fragmented, leading to missing morphological information. Currently, the reassembly of fragmented objects from a collection of hundreds of mixed fragments is manually done by experts. To overcome the pitfalls of manual reconstruction and improve the quality of reconstructed samples, here we present a Generative Adversarial Network (GAN) framework tested on an extensive database with complete and fragmented references. Using customized GANs, we trained a model with 1072 samples corresponding to Iberian wheel-made pottery profiles belonging to archaeological sites located in the upper valley of the Guadalquivir River (Spain). We provide quantitative and qualitative assessments to measure the quality of the reconstructed samples, along with domain expert evaluation with archaeologists. The resulting framework is a possible way to facilitate pottery reconstruction from partial fragments of an original piece.

[![CODE](https://img.shields.io/badge/CODE-73ff9.svg)](https://github.com/celiacintas/vasijas/tree/iberianGAN)


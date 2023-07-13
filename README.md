# Reconstruction of Iberian ceramic potteries using auto-encoder generative adversarial networks
[![DOI](https://img.shields.io/badge/DOI-10.1038/s41598.022.14910.7-f9f107.svg)](https://doi.org/10.1038/s41598-022-14910-7)

## Abstract
Several aspects of past culture, including historical trends, are inferred from time-based patterns observed in archaeological artifacts belonging to different periods. Because its presence and variation give important clues about the Neolithic revolution and given their relative abundance in most archaeological sites, ceramic potteries are significantly helpful in this purpose. Nonetheless, most available pottery is fragmented, leading to missing morphological information. Currently, the reassembly of fragmented objects from a collection of hundreds of mixed fragments is manually done by experts. To overcome the pitfalls of manual reconstruction and improve the quality of reconstructed samples, here we present a Generative Adversarial Network (GAN) framework tested on an extensive database with complete and fragmented references. Using customized GANs, we trained a model with 1072 samples corresponding to Iberian wheel-made pottery profiles belonging to archaeological sites located in the upper valley of the Guadalquivir River (Spain). We provide quantitative and qualitative assessments to measure the quality of the reconstructed samples, along with domain expert evaluation with archaeologists. The resulting framework is a possible way to facilitate pottery reconstruction from partial fragments of an original piece. 

## Dataset

The raw data belong to binary profile images, corresponding to Iberian wheel-made pottery from various archaeological sites of the upper valley of the Guadalquivir River (Spain). The available images consist of a profile view of the pottery, where image resolutions (in pixels), corresponding to size scale, may vary according to the acquisition settings. We partitioned these images into rim and base portion to simulate the fractures in the profiles. The partitioning criterion and orientation depends on the initial shape (closed or open).


[![DOI](https://img.shields.io/badge/LINK_DATASET-f9f107.svg)](https://drive.google.com/file/d/11BfsJQocyZyHcx53IbsZAJFcHasW5yNR/view?usp=sharing)


![DATASET](fig/dataset.png)

## Overview
![DATASET](fig/overview.png)

## Results
![DATASET](fig/result.png)

## More Papers

- IberianVoxel: Automatic Completion of Iberian Ceramics for Cultural Heritage Studies

[![CODE](https://camo.githubusercontent.com/cab0ba8ebc53130e4e17ecf07c91c58c3d369da13fd2b4dabfb495be044a5c6c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f434f44452d37336666392e737667)](https://github.com/celiacintas/vasijas/tree/iberianVox)

- Learning feature representation of Iberian ceramics with automatic classification models

[![DOI](https://img.shields.io/badge/DOI-10.1016/j.culher.2021.01.003-f9f107.svg)](https://doi.org/10.1016/j.culher.2021.01.003)
[![CODE](https://img.shields.io/badge/CODE-73ff9.svg)](https://github.com/celiacintas/vasijas/tree/unsupervised)

- Automatic feature extraction and classification of Iberian ceramics based on deep convolutional networks

[![DOI](https://img.shields.io/badge/DOI-10.1016/j.culher.2019.06.005-f9f107.svg)](https://doi.org/10.1016/j.culher.2019.06.005)
[![CODE](https://img.shields.io/badge/CODE-73ff9.svg)](https://github.com/celiacintas/vasijas/tree/classification)

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




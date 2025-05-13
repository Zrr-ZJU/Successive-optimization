# Successive optimization of optics and post-processing with differentiable coherent PSF operator and field information

### Zheng Ren, Jingwen Zhou, Wenguan Zhang, Jiapu Yan, Bingkun Chen, Huajun Feng, Shiqi Chen

This is the official code repository for the paper: "Successive optimization of optics and post-processing with differentiable coherent PSF operator and field information".

<div style="text-align:center;">
    <img src="joint overview.jpg"/>
</div>

## Key contributions 
- Using a **customized differential operator** that efficiently arrange the memory, we develop a differentiable optical simulation model that avoids exponentially growing computation overhead and could accurately calculates coherent PSFs;
- A joint optimization pipeline is presented that **not only enhances image quality**, but also **successively improves the performance of optics** across multiple lenses that are already in professional level;
- For the first time, we show that joint optimization could realize a **field-level PSF control** in advanced optical design, revealing its tremendous potential by bringing evaluated lenses approaching the diffraction limit with an improved effective modulation transfer function (EMTF).

## Citation
```
@ARTICLE{10976391,
  author={Ren, Zheng and Zhou, Jingwen and Zhang, Wenguan and Yan, Jiapu and Chen, Bingkun and Feng, Huajun and Chen, Shiqi},
  journal={IEEE Transactions on Computational Imaging}, 
  title={Successive Optimization of Optics and Post-processing with Differentiable Coherent PSF Operator and Field Information}, 
  year={2025},
  volume={},
  number={},
  pages={1-10},
  keywords={Optical diffraction;Optical imaging;Adaptive optics;Lenses;Optimization;Optical refraction;Optical design;Optical surface waves;Apertures;Ray tracing;Joint lens design;differentiable optical simulation;memory-efficient backpropagation;image reconstruction},
  doi={10.1109/TCI.2025.3564173}}
```

## Note (further improving efficiency and convenience)
- Merging wavelength and field batches can speed up calculations
- Use scripts to read in lens json files from zmx files and materials from AGF files
#### A more concise and efficient version may be updated.
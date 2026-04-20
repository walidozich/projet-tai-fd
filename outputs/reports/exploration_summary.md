# Exploration des donnees

Cette exploration utilise les fichiers charges avec alignement en memoire : les images sont converties en RGB, les masques sont alignes sur la premiere image de la scene avec nearest-neighbor si necessaire.

## Resume numerique

| Scene | Image shapes | Mask shape | Image dtype | Mask dtype | RGB min/max | Mask unique count | Mask type observed |
|---|---|---|---|---|---|---:|---|
| scene1 | (486, 558, 3) | (486, 558, 4) | uint8 | uint8 | [1, 2, 0] / [255, 255, 255] | 30549 | multi-channel mask |
| scene2 | (349, 366, 3) | (349, 366, 4) | uint8 | uint8 | [0, 0, 0] / [255, 255, 255] | 842 | multi-channel mask |
| scene3 | (411, 411, 3) | (411, 411, 4) | uint8 | uint8 | [13, 26, 11] / [255, 255, 255] | 6363 | multi-channel mask |
| scene4 | (526, 541, 3), (526, 541, 3) | (526, 541) | uint8 | uint8 | [0, 0, 0] / [255, 255, 255] | 2 | binary |

## Details par scene

### scene1

- Objectif : Segmenter chat, ciel, sol, arbres.
- Dimensions image(s) chargee(s) : (486, 558, 3).
- Dimensions masque charge : (486, 558, 4).
- Type image : uint8; type masque : uint8.
- Min RGB : [1, 2, 0]; max RGB : [255, 255, 255].
- Nombre de valeurs/couleurs uniques dans le masque : 30549.
- Echantillon de valeurs du masque : `[[90, 135, 223, 255], [90, 136, 225, 255], [91, 135, 222, 255], [91, 135, 223, 255], [91, 136, 224, 255], [91, 136, 225, 255], [92, 135, 221, 255], [92, 135, 222, 255], [92, 136, 221, 255], [92, 136, 222, 255]]`.

### scene2

- Objectif : Extraire le disque lumineux.
- Dimensions image(s) chargee(s) : (349, 366, 3).
- Dimensions masque charge : (349, 366, 4).
- Type image : uint8; type masque : uint8.
- Min RGB : [0, 0, 0]; max RGB : [255, 255, 255].
- Nombre de valeurs/couleurs uniques dans le masque : 842.
- Echantillon de valeurs du masque : `[[0, 0, 0, 255], [0, 0, 1, 255], [0, 0, 2, 255], [0, 1, 0, 255], [0, 1, 1, 255], [0, 1, 2, 255], [0, 1, 3, 255], [0, 2, 0, 255], [0, 2, 1, 255], [0, 2, 2, 255]]`.

### scene3

- Objectif : Isoler les routes.
- Dimensions image(s) chargee(s) : (411, 411, 3).
- Dimensions masque charge : (411, 411, 4).
- Type image : uint8; type masque : uint8.
- Min RGB : [13, 26, 11]; max RGB : [255, 255, 255].
- Nombre de valeurs/couleurs uniques dans le masque : 6363.
- Echantillon de valeurs du masque : `[[0, 0, 0, 255], [0, 0, 1, 255], [0, 0, 2, 255], [0, 0, 3, 255], [0, 0, 4, 255], [0, 0, 5, 255], [0, 0, 6, 255], [0, 0, 7, 255], [0, 0, 8, 255], [0, 0, 9, 255]]`.

### scene4

- Objectif : Extraire la personne debout.
- Dimensions image(s) chargee(s) : (526, 541, 3), (526, 541, 3).
- Dimensions masque charge : (526, 541).
- Type image : uint8; type masque : uint8.
- Min RGB : [0, 0, 0]; max RGB : [255, 255, 255].
- Nombre de valeurs/couleurs uniques dans le masque : 2.
- Echantillon de valeurs du masque : `[0, 255]`.

## Notes de difficulte initiales

- Scene 1 : segmentation multi-classes. Le masque Ground Truth est multi-canal avec beaucoup de couleurs uniques, donc une normalisation des labels sera necessaire avant les metriques multi-classes.
- Scene 2 : extraction binaire du disque lumineux. Le masque est multi-canal avec beaucoup de valeurs uniques, probablement a cause d anti-aliasing ou d une image non strictement binaire ; il faudra binariser proprement avant evaluation.
- Scene 3 : les routes sont fines et peuvent etre confondues avec d autres structures claires. Le masque contient beaucoup de valeurs uniques, donc une binarisation/normalisation sera necessaire.
- Scene 4 : la scene contient une image RGB et une image supplementaire. Le masque charge est binaire 0/255, ce qui simplifie l evaluation, mais l image supplementaire doit rester alignee sur l image RGB.

## Figures generees

- `outputs/visualisations/scene1_image_mask_overview.png`
- `outputs/visualisations/scene1_intensity_histogram.png`
- `outputs/visualisations/scene1_rgb_histogram.png`
- `outputs/visualisations/scene2_image_mask_overview.png`
- `outputs/visualisations/scene2_intensity_histogram.png`
- `outputs/visualisations/scene2_rgb_histogram.png`
- `outputs/visualisations/scene3_image_mask_overview.png`
- `outputs/visualisations/scene3_intensity_histogram.png`
- `outputs/visualisations/scene3_rgb_histogram.png`
- `outputs/visualisations/scene4_image_mask_overview.png`
- `outputs/visualisations/scene4_intensity_histogram.png`
- `outputs/visualisations/scene4_rgb_histogram.png`

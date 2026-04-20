# Projet TAI-FD - Segmentation d'Images

Ce projet vise a construire une chaine complete de segmentation d'objets dans 4 scenes complexes en utilisant uniquement les notions vues en TAI et FD.

## Approche retenue

Le projet utilisera une approche hybride :

- un notebook pour l'exploration, les visualisations, les essais et la presentation des resultats ;
- des modules Python dans `src/` pour les fonctions reutilisables ;
- un dossier `outputs/` pour les masques predits, figures, metriques et rapports.

## Donnees detectees

Les fichiers actuels dans `dataset/` sont :

- `Scene_1.png` avec `GT1.png`
- `Scene_2.png` avec `GT2.png`
- `Scene_3.png` avec `GT3.png`
- `Scene_4_RGB_1.png` et `Scene_4_D_2.png` avec `GT4.png`

## Structure

```text
notebooks/projet_segmentation.ipynb
src/preprocessing.py
src/segmentation_tai.py
src/segmentation_fd.py
src/evaluation.py
src/visualization.py
src/main.py
outputs/masks_predits/
outputs/visualisations/
outputs/metrics/
outputs/reports/
```

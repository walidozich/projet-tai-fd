# TODO - Projet TAI-FD Segmentation d'Images

## Decision sur le format du projet

- [x] Utiliser un notebook pour l'exploration, les tests visuels et la redaction progressive des resultats.
- [x] Garder aussi une structure de code Python modulaire pour eviter un projet uniquement sous forme de cellules difficiles a maintenir.
- [x] Format recommande : approche hybride.
- [x] Notebook principal : `notebooks/projet_segmentation.ipynb` pour les experiences, visualisations, tableaux et commentaires.
- [x] Scripts Python : `src/` pour les fonctions reutilisables de preprocessing, segmentation, evaluation et visualisation.
- [ ] Rapport final : exporter le notebook en PDF/HTML ou reprendre les resultats dans un fichier Markdown/PDF.

## Structure du projet

- [x] Creer/verifier la structure suivante :

```text
Projet TAI-FD/
├── dataset/
│   ├── Scene_1.png
│   ├── Scene_2.png
│   ├── Scene_3.png
│   ├── Scene_4_RGB_1.png
│   ├── Scene_4_D_2.png
│   ├── GT1.png
│   ├── GT2.png
│   ├── GT3.png
│   └── GT4.png
├── notebooks/
│   └── projet_segmentation.ipynb
├── outputs/
│   ├── masks_predits/
│   ├── visualisations/
│   ├── metrics/
│   └── reports/
├── src/
│   ├── preprocessing.py
│   ├── segmentation_tai.py
│   ├── segmentation_fd.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── main.py
├── README.md
└── todo.md
```

- [x] Creer les dossiers manquants.
- [x] Verifier les noms exacts des fichiers images et masques dans `dataset/`.
- [x] Decider une convention claire pour les sorties : `scene1_kmeans_mask.png`, `scene2_otsu_mask.png`, etc.

## Installation et environnement

- [x] Creer un environnement Python si necessaire. Current environment is sufficient.
- [x] Installer les bibliotheques utiles : `numpy`, `pandas`, `matplotlib`, `opencv-python`, `scikit-learn`, `scipy`, `Pillow`, `jupyter`. Already available.
- [x] Verifier si `scikit-learn-extra` est disponible pour K-Medoids. Available.
- [x] Si K-Medoids n'est pas disponible, implementer une version simple PAM ou expliquer la limitation. Not needed because `scikit-learn-extra` is available.
- [x] Creer un fichier `requirements.txt`.
- [x] Tester l'import de toutes les bibliotheques.

## Chargement du dataset

- [x] Telecharger le dataset depuis le lien Google Drive fourni dans l'enonce. Dataset already present in `dataset/`.
- [x] Placer les donnees dans `dataset/`.
- [x] Identifier les 4 scenes.
- [x] Identifier les masques Ground Truth correspondants.
- [x] Pour la scene 4, identifier l'image couleur et l'image contenant l'information supplementaire.
- [x] Verifier que chaque image a un masque associe.
- [x] Verifier que les dimensions image/masque sont compatibles. Raw mismatches found and documented.
- [x] Redimensionner uniquement si necessaire, en gardant la coherence avec le masque. Implemented in-memory alignment: bilinear for extra images, nearest-neighbor for masks.
- [x] Sauvegarder un resume du dataset dans `outputs/reports/dataset_summary.md`.

## Exploration des donnees

- [ ] Afficher chaque image originale.
- [ ] Afficher chaque masque Ground Truth.
- [ ] Afficher les dimensions : hauteur, largeur, nombre de canaux.
- [ ] Afficher le type de donnees : `uint8`, `float`, etc.
- [ ] Afficher les valeurs min/max par canal.
- [ ] Afficher les valeurs uniques dans les masques.
- [ ] Verifier si les masques sont binaires ou multi-classes.
- [ ] Tracer les histogrammes des intensites pour chaque scene.
- [ ] Tracer les histogrammes RGB si utile.
- [ ] Noter les difficultes visuelles de chaque scene.

## Fonctions communes a implementer

- [ ] Fonction `load_image(path, rgb=True)`.
- [ ] Fonction `load_mask(path)`.
- [ ] Fonction `save_mask(mask, path)`.
- [ ] Fonction `show_image_mask(image, mask, title)`.
- [ ] Fonction `overlay_mask(image, mask)`.
- [ ] Fonction `normalize_minmax(features)`.
- [ ] Fonction `standardize_zscore(features)`.
- [ ] Fonction `resize_for_clustering(image, max_size)` si les images sont trop grandes.
- [ ] Fonction `extract_rgb_features(image)`.
- [ ] Fonction `extract_rgb_xy_features(image)`.
- [ ] Fonction `extract_gray_gradient_xy_features(image)`.
- [ ] Fonction `remove_small_components(mask, min_area)`.
- [ ] Fonction `keep_largest_component(mask)`.

## Pretraitement TAI

- [ ] Conversion BGR vers RGB si OpenCV est utilise.
- [ ] Conversion RGB vers niveaux de gris.
- [ ] Calcul d'histogramme.
- [ ] Calcul d'histogramme cumule.
- [ ] Normalisation d'histogramme si utile.
- [ ] Filtre moyenneur.
- [ ] Filtre gaussien avec parametre sigma/kernel.
- [ ] Filtre median pour bruit impulsionnel.
- [ ] Sobel horizontal et vertical.
- [ ] Magnitude du gradient Sobel.
- [ ] Direction du gradient si necessaire.
- [ ] Seuillage simple.
- [ ] Seuillage d'Otsu.
- [ ] Composantes connexes en 8-connectivite.
- [ ] Flood fill si utile pour region growing.

## Methodes FD a implementer/utiliser

- [ ] K-Means.
- [ ] K-Medoids / PAM.
- [ ] AGNES : clustering hierarchique agglomeratif.
- [ ] DIANA : clustering divisif ou approximation documentee si implementation complete trop lourde.
- [ ] DBSCAN.
- [ ] Methode du coude pour choisir `k`.
- [ ] Detection automatique du coude si possible.
- [ ] Calcul de distances pairwise si necessaire.
- [ ] Calcul de l'inertie/SSE.
- [ ] PCA 2D pour visualiser les clusters.
- [ ] Silhouette score.
- [ ] Davies-Bouldin score.
- [ ] Calinski-Harabasz score.

## Scene 1 - Chat, ciel, sol, arbres

Objectif : segmenter l'image en 4 clusters distincts.

- [ ] Charger image + masque Ground Truth.
- [ ] Afficher image et masque.
- [ ] Appliquer un leger flou gaussien.
- [ ] Extraire les features `[R, G, B]`.
- [ ] Extraire les features `[R, G, B, x, y]`.
- [ ] Normaliser les features.
- [ ] Appliquer K-Means avec `k = 4`.
- [ ] Visualiser les 4 clusters.
- [ ] Associer chaque cluster a une classe : chat, ciel, sol, arbres.
- [ ] Nettoyer les petites regions avec composantes connexes.
- [ ] Tester K-Medoids/PAM sur image reduite si necessaire.
- [ ] Tester AGNES sur image reduite.
- [ ] Comparer les methodes.
- [ ] Calculer les metriques multi-classes : accuracy, precision par classe, recall par classe, F1 par classe, IoU par classe, mean IoU.
- [ ] Calculer les metriques internes : silhouette, Davies-Bouldin, Calinski-Harabasz, inertie.
- [ ] Sauvegarder masque predit et visualisations.
- [ ] Ecrire une interpretation courte des resultats.

## Scene 2 - Disque lumineux

Objectif : extraire la zone lumineuse circulaire.

- [ ] Charger image + masque Ground Truth.
- [ ] Convertir en niveaux de gris.
- [ ] Afficher l'histogramme d'intensite.
- [ ] Tester une normalisation d'histogramme si contraste faible.
- [ ] Appliquer un filtre gaussien.
- [ ] Appliquer Otsu.
- [ ] Appliquer un seuillage simple manuel pour comparaison.
- [ ] Utiliser les composantes connexes.
- [ ] Garder la composante correspondant au disque lumineux.
- [ ] Supprimer les petites regions parasites.
- [ ] Tester K-Means avec `k = 2` sur l'intensite.
- [ ] Comparer Otsu, seuil simple, K-Means.
- [ ] Calculer IoU, Dice/F1, precision, recall, accuracy.
- [ ] Sauvegarder masque predit et visualisations.
- [ ] Ecrire une interpretation courte des erreurs : sous-segmentation, sur-segmentation, bruit.

## Scene 3 - Routes en vue aerienne

Objectif : isoler exclusivement les routes.

- [ ] Charger image + masque Ground Truth.
- [ ] Afficher image et masque.
- [ ] Analyser les couleurs des routes par rapport au fond.
- [ ] Tester flou gaussien ou median.
- [ ] Calculer le gradient Sobel.
- [ ] Construire features `[R, G, B]`.
- [ ] Construire features `[R, G, B, intensite, gradient, x, y]`.
- [ ] Normaliser les features.
- [ ] Tester K-Means avec `k = 3`, `k = 4`, `k = 5`.
- [ ] Utiliser la methode du coude pour justifier `k`.
- [ ] Identifier le cluster correspondant aux routes.
- [ ] Nettoyer avec composantes connexes.
- [ ] Supprimer les petites regions non-route.
- [ ] Tester DBSCAN sur pixels candidats ou image reduite.
- [ ] Tester une methode TAI simple : seuillage intensite/couleur si possible.
- [ ] Comparer les methodes.
- [ ] Calculer IoU, Dice/F1, precision, recall, accuracy.
- [ ] Mettre l'accent sur recall si l'objectif est de retrouver un maximum de routes.
- [ ] Sauvegarder masque predit et visualisations.
- [ ] Ecrire une interpretation : routes manquees, batiments confondus, vegetation confondue.

## Scene 4 - Extraction de la personne debout

Objectif : extraire uniquement la personne en utilisant l'image couleur et l'information supplementaire.

- [ ] Charger image couleur.
- [ ] Charger image supplementaire.
- [ ] Charger masque Ground Truth.
- [ ] Verifier que les images sont alignees.
- [ ] Verifier que les dimensions correspondent.
- [ ] Afficher les deux images et le masque.
- [ ] Analyser le role de l'image supplementaire : profondeur, carte thermique, contraste, autre.
- [ ] Tester Otsu sur l'image supplementaire.
- [ ] Tester seuil simple sur l'image supplementaire.
- [ ] Construire features `[R, G, B, extra]`.
- [ ] Construire features `[R, G, B, extra, x, y]`.
- [ ] Normaliser les features.
- [ ] Appliquer K-Means avec plusieurs valeurs de `k`.
- [ ] Tester DBSCAN si l'image supplementaire separe bien la personne.
- [ ] Identifier le cluster correspondant a la personne.
- [ ] Utiliser composantes connexes pour garder uniquement la personne debout.
- [ ] Supprimer les autres objets/personnes si presents.
- [ ] Calculer IoU, Dice/F1, precision, recall, accuracy.
- [ ] Mettre l'accent sur precision pour eviter d'inclure le fond.
- [ ] Sauvegarder masque predit et visualisations.
- [ ] Ecrire une interpretation : parties manquees, fond inclus, effet de l'image supplementaire.

## Evaluation des performances

- [ ] Implementer la matrice de confusion binaire : TP, TN, FP, FN.
- [ ] Implementer accuracy.
- [ ] Implementer precision.
- [ ] Implementer recall.
- [ ] Implementer F1-score / Dice.
- [ ] Implementer IoU / Jaccard.
- [ ] Implementer evaluation multi-classes pour scene 1.
- [ ] Implementer IoU par classe.
- [ ] Implementer mean IoU.
- [ ] Gerer les divisions par zero proprement.
- [ ] Verifier que les masques predits et Ground Truth ont les memes labels.
- [ ] Verifier que les masques binaires utilisent bien 0/1 ou 0/255 de maniere coherente.
- [ ] Sauvegarder les metriques en CSV.
- [ ] Sauvegarder les metriques en JSON.
- [ ] Produire un tableau comparatif final par scene.

## Visualisations a produire

- [ ] Image originale.
- [ ] Masque Ground Truth.
- [ ] Masque predit.
- [ ] Superposition du masque predit sur l'image originale.
- [ ] Comparaison cote a cote : original / GT / prediction / overlay.
- [ ] Histogrammes avant/apres preprocessing si pertinent.
- [ ] Courbe du coude pour K-Means.
- [ ] PCA 2D des features colorees par cluster.
- [ ] Matrice de confusion.
- [ ] Sauvegarder toutes les figures dans `outputs/visualisations/`.

## Comparaison des methodes

- [ ] Pour chaque scene, tester au moins une methode TAI et une methode FD quand c'est pertinent.
- [ ] Pour chaque scene, identifier la meilleure methode selon IoU/Dice.
- [ ] Comparer les avantages et limites des methodes TAI.
- [ ] Comparer les avantages et limites des methodes FD.
- [ ] Expliquer pourquoi une methode fonctionne mieux sur une scene donnee.
- [ ] Documenter les cas d'echec.
- [ ] Ne pas utiliser de methodes hors programme comme reseaux de neurones, U-Net, YOLO, SAM, segmentation deep learning.

## Rapport final

- [ ] Rediger une introduction.
- [ ] Presenter l'objectif du projet.
- [ ] Decrire le dataset.
- [ ] Decrire les notions TAI utilisees.
- [ ] Decrire les notions FD utilisees.
- [ ] Presenter le pipeline general.
- [ ] Presenter les resultats de la scene 1.
- [ ] Presenter les resultats de la scene 2.
- [ ] Presenter les resultats de la scene 3.
- [ ] Presenter les resultats de la scene 4.
- [ ] Inclure les tableaux de metriques.
- [ ] Inclure les visualisations principales.
- [ ] Discuter les erreurs observees.
- [ ] Comparer TAI vs FD.
- [ ] Conclure avec la meilleure approche par scene.
- [ ] Mentionner les limites du travail.

## README

- [ ] Expliquer l'objectif du projet.
- [ ] Expliquer comment organiser le dataset.
- [ ] Expliquer comment installer les dependances.
- [ ] Expliquer comment lancer le notebook.
- [ ] Expliquer comment lancer les scripts si disponibles.
- [ ] Expliquer ou sont sauvegardes les resultats.
- [ ] Lister les principales methodes utilisees.

## Verification finale

- [ ] Le projet respecte exclusivement les concepts vus en TP TAI et FD.
- [ ] Les 4 scenes sont traitees.
- [ ] Chaque scene a au moins un masque predit sauvegarde.
- [ ] Chaque scene a des metriques calculees.
- [ ] Chaque scene a une visualisation comparative.
- [ ] Les resultats sont reproductibles.
- [ ] Le notebook s'execute du debut a la fin sans erreur.
- [ ] Les chemins relatifs fonctionnent.
- [ ] Le rapport contient les figures et tableaux necessaires.
- [ ] Les conclusions sont basees sur les metriques, pas seulement sur l'observation visuelle.

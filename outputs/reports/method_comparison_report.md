# Comparaison des Methodes

Ce rapport compare les methodes TAI et FD testees sur les quatre scenes. Les methodes utilisees restent limitees aux notions vues en TP : seuillage, filtrage, composantes connexes, K-Means, K-Medoids/PAM, AGNES, DBSCAN, PCA et metriques de clustering/segmentation.

## Resume des meilleures methodes

| Scene | Meilleure methode | Type | Score principal | Interpretation courte |
| --- | --- | --- | ---: | --- |
| scene1 | `kmeans_rgb_xy_k4` | FD | mean IoU = 0.5057 | K-Means respecte la demande de 4 clusters et les coordonnees XY aident la coherence spatiale. |
| scene2 | `high_percentile_94_largest_component` | TAI | IoU = 0.2100 | Le disque est une zone tres lumineuse ; un seuil fort est plus adapte qu Otsu global. |
| scene3 | `tai_hsv_s20_v150_open` | TAI | IoU = 0.2319 | Les routes sont gris/desaturees ; le seuillage HSV donne moins de faux positifs que le clustering. |
| scene4 | `kmeans_rgb_extra_xy_k5` | FD | IoU = 0.6935 | L image supplementaire separe la personne ; K-Means avec extra+XY isole bien la silhouette. |

## Comparaison TAI vs FD

### Methodes TAI

Avantages :

- Simples, rapides et interpretables.
- Tres efficaces quand l objet a une propriete visuelle claire : intensite, couleur, saturation, composante connexe dominante.
- Donnent de bons resultats sur Scene 2 et Scene 3 quand une hypothese image claire existe.

Limites :

- Sensibles aux seuils choisis.
- Otsu peut echouer si l histogramme est domine par un grand fond non pertinent.
- Les seuils couleur/intensite confondent facilement routes, batiments, sol clair ou reflets.
- Les contours et gradients ne suffisent pas seuls a isoler un objet semantique.

### Methodes FD

Avantages :

- Permettent de combiner plusieurs features : couleur, intensite, gradient, position XY, image supplementaire.
- K-Means fonctionne bien quand le nombre de groupes est connu ou impose.
- Les coordonnees XY ameliorent souvent la coherence spatiale des clusters.
- L image supplementaire de Scene 4 devient tres exploitable dans un vecteur de features.

Limites :

- Les clusters ne correspondent pas toujours a des classes semantiques.
- Il faut choisir ou identifier le bon cluster apres coup par heuristique image.
- DBSCAN depend fortement de `eps` et `min_samples`.
- AGNES/K-Medoids deviennent couteux sur tous les pixels, donc ils sont testes sur images reduites ou echantillons.

## Analyse par scene

### Scene 1

- Objectif : segmenter en 4 classes : chat, ciel, sol, arbres.
- Methodes comparees : K-Means RGB, K-Means RGB+XY, K-Medoids/PAM sur echantillon, AGNES sur echantillon.
- Meilleure methode retenue : K-Means RGB+XY avec `k=4`.
- Pourquoi : l objectif impose 4 clusters et les coordonnees XY reduisent les melanges spatiaux entre regions de couleurs proches.
- Limite : les labels de cluster ne sont pas semantiques ; l association chat/ciel/sol/arbres reste heuristique. Le GT a beaucoup de couleurs et doit etre normalise en 4 labels.

### Scene 2

- Objectif : extraire le disque lumineux.
- Methodes comparees : Otsu, seuil fixe, seuil percentile fort, K-Means intensite `k=2`.
- Meilleure methode retenue : seuil percentile 94 + plus grande composante connexe.
- Pourquoi : le disque est une zone tres lumineuse locale. Otsu et K-Means `k=2` segmentent trop largement car le fond noir influence fortement la distribution globale.
- Limite : les vaisseaux et variations locales coupent la zone lumineuse, donc le masque reste imparfait.

### Scene 3

- Objectif : isoler les routes.
- Methodes comparees : seuillage HSV, K-Means sur features route, DBSCAN sur pixels candidats.
- Meilleure methode retenue : HSV faible saturation / forte valeur + ouverture morphologique.
- Pourquoi : les routes sont en general grises et peu saturees. Un seuil strict limite les faux positifs.
- Limite : beaucoup de routes sont manquees. Les methodes plus larges ameliorent le recall mais ajoutent des batiments/toits clairs.

### Scene 4

- Objectif : extraire la personne debout avec image RGB + image supplementaire.
- Methodes comparees : Otsu sur extra, seuil simple sur extra, K-Means RGB+extra, K-Means RGB+extra+XY, DBSCAN.
- Meilleure methode retenue : K-Means RGB+extra+XY avec `k=5`.
- Pourquoi : l image supplementaire donne une silhouette mid-gray de la personne et XY aide a garder la region coherente debout.
- Limite : erreurs de bord autour des bras, jambes et zones bruitees dans l image supplementaire.

## Cas d'echec documentes

- Scene 1 : GT non propre en 4 labels ; comparaison numerique depend de la normalisation du GT.
- Scene 2 : Otsu/K-Means intensite confondent fond noir et champ retinien, puis selectionnent une region trop grande.
- Scene 3 : routes confondues avec batiments clairs ; meilleur IoU obtenu avec une methode precise mais faible recall.
- Scene 4 : Otsu inverse inclut trop de fond ; DBSCAN a bon recall mais precision faible.

## Verification des contraintes

- Aucun reseau de neurones n est utilise.
- Aucune methode hors programme comme U-Net, YOLO ou SAM n est utilisee.
- Les masques Ground Truth sont utilises uniquement apres prediction pour evaluation.
- Les methodes restent dans le cadre TAI/FD vu en TP.

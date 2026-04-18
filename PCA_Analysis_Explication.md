# Comprendre la "PCA Analysis" (Analyse en Composantes Principales) sur la Courbe des Taux

Ce document explique les concepts et l'usage en gestion de portefeuille du module d'Analyse en Composantes Principales (PCA - *Principal Component Analysis*) implémenté dans notre projet (fichier `pca_analysis.py`).

## 1. Qu'est-ce que la PCA appliquée à la courbe des taux ?

En finance de marché, et particulièrement dans l'obligataire (Fixed Income), la courbe des taux (Yield Curve) bouge tous les jours. Ces mouvements peuvent sembler chaotiques car chaque maturité (1 an, 5 ans, 10 ans...) évolue de manière légèrement différente.

Cependant, historiquement, on observe que les mouvements de l'ensemble de ces points sont fortement corrélés. **L'Analyse en Composantes Principales (PCA)** est une technique statistique qui permet de résumer ces dizaines de mouvements complexes en seulement **3 mouvements fondamentaux** (les composantes principales), qui expliquent à eux seuls plus de 95 % des variations historiques de la courbe.

## 2. Les 3 Mouvements Fondamentaux (Facteurs PCA)

Dans notre code (`get_default_pca_loadings`), nous avons calibré les "loadings" (poids ou sensibilités) historiques de chaque maturité face à ces 3 facteurs :

### 🔸 PC1 : Le Niveau (Level)
* **Poids dans la variance** : ~85-90% de l'explication des mouvements.
* **Que se passe-t-il ?** : C'est un mouvement de **translation parallèle**. Toute la courbe monte ou descend en même temps. Les taux courts, moyens et longs bougent dans la même direction et avec des amplitudes similaires.
* **Impact sur le portefeuille** : C'est le risque directionnel classique des taux (la Duration).

### 🔸 PC2 : La Pente (Slope)
* **Poids dans la variance** : ~5-10%.
* **Que se passe-t-il ?** : C'est un mouvement de **Pentification (Steepening) ou d'Aplatissement (Flattening)** de la courbe.
  * *Exemple* : Les taux à court terme baissent (loadings négatifs) pendant que les taux à long terme montent (loadings positifs). La courbe devient plus "raide".

### 🔸 PC3 : La Courbure (Curvature / Butterfly)
* **Poids dans la variance** : ~1-3%.
* **Que se passe-t-il ?** : C'est un mouvement du **"ventre" (belly) de la courbe par rapport aux "ailes" (wings)**.
  * *Exemple* : Les taux à 5 ans baissent (le ventre creuse), mais les taux à 3 mois et à 30 ans montent (les ailes se soulèvent). On appelle cela un mouvement "Papillon" (Butterfly).

## 3. L'Usage en Gestion de Portefeuille

Pourquoi un gérant de fonds (Portfolio Manager) utilise-t-il ce module ?

Au lieu de regarder son exposition au risque maturité par maturité (ce qui est illisible s'il a 500 lignes), le gérant veut savoir : *"Comment mon portefeuille va-t-il réagir si la courbe se déplace, si elle se pentifie, ou si elle se tord au milieu ?"*

### Le Calcul (Fonction `decompose_portfolio_pca_exposure`)
1. Le gestionnaire calcule le risque en dollars sur chaque "point" de la courbe pour son portefeuille. C'est le **DV01 par Tenor** (Dollar Value of 1 bp : combien il gagne/perd si le taux local bouge de 1 point de base).
2. L'algorithme vient multiplier ce DV01 par les sensibilités (loadings) PCA correspondantes.
3. Il somme le tout pour obtenir trois "méta-expositions" en équivalent-DV01 : l'exposition au Niveau, à la Pente et à la Courbure.

### Interprétation des Signaux (`interpret_pca_exposures`)

Une fois les expositions calculées, le module donne des indications claires pour piloter le portefeuille :

* **Level Exposure (Exposition à la Duration)** :
  * **Positif (>0)** : Le portefeuille est "Long Duration". Il gagnera de l'argent si la banque centrale baisse les taux (translation vers le bas).
  * **Négatif (<0)** : Le portefeuille est "Short Duration". Il gagnera si les taux montent de manière globale.

* **Slope Exposure (Exposition à la Pente)** :
  * **Positif (>0)** : Le portefeuille est "Long Slope" (Position pentificatrice). Le gérant parie que la différence entre les taux longs et les taux courts va augmenter (Steepening).
  * **Négatif (<0)** : Le portefeuille est "Short Slope". Le gérant parie sur un aplatissement de la courbe (Flattening).

* **Curvature Exposure (Exposition à la Courbure)** :
  * **Positif (>0)** : Sous-pondération du "ventre" (ex: 5 ans) et surpondération des "ailes" (ex: 2 ans et 10 ans). Le pari est gagnant si le ventre de la courbe monte proportionnellement plus vite que les extrémités.
  * **Négatif (<0)** : Surpondération du "ventre". On parie que la courbe va s'arrondir ou creuser son milieu.

---

### Résumé de l'utilité pour le métier
La PCA est l'outil diagnostique incontournable pour les gérants "Macro". Elle leur permet de :
1. **Traduire** des positions très complexes de centaines d'obligations en 3 paris clairs et simples à comprendre (Level, Slope, Curvature).
2. **Couvrir (Hedger)** le portefeuille efficacement : En connaissant ces 3 sensibilités, le gérant sait exactement quels contrats à terme (Futures) acheter ou vendre pour immuniser le portefeuille contre certaines déformations de la courbe.

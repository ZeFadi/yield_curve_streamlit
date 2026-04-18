# Comprendre le "Relative Value : Rich / Cheap Screening"

Ce document explique les concepts, les mathématiques et la logique derrière le module de *Relative Value* (valeur relative) implémenté dans notre projet, en particulier la détection des obligations "Rich" (Chères) ou "Cheap" (Bon marché). L'implémentation se trouve principalement dans le fichier `relative_value.py`.

## 1. Le Z-Spread (Zero-Volatility Spread)

### Qu'est-ce que c'est ?
Le Z-Spread est la marge (ou "spread") constante en points de base (bps) qu'il faut ajouter uniformément à l'ensemble de la courbe des taux zéro-coupon (Yield Curve) de référence, pour que la valeur actualisée (Present Value - PV) des flux de trésorerie futurs d'une obligation soit exactement égale à son prix de marché observé.

### La Formule & Résolution mathématique
L'équation que l'algorithme cherche à résoudre pour trouver le Z-Spread est :

$$ PV(Z\text{-}Spread) = P_{\text{clean}} + AI $$

*où :*
* $PV$ : La fonction qui actualise les flux futurs.
* $P_{\text{clean}}$ : Le prix de marché de l'obligation observé (en % du nominal).
* $AI$ : Les intérêts courus (Accrued Interest).

Dans le code (fonction `compute_z_spread`), il n'existe pas de formule explicite (fermée) pour extraire ce spread. Nous utilisons donc un **algorithme de recherche de racine** (la méthode de Brent, via `scipy.optimize.brentq`). L'algorithme teste itérativement des spreads dans un intervalle large (de -1000 bps à +5000 bps) jusqu'à ce que la différence entre la valorisation théorique et le prix du marché réel soit négligeable (proche de zéro).

## 2. La Régression : Le Spread Modélisé ("Fitted Spread")

### Le rationnel
Une fois que nous avons calculé le Z-Spread et la sensibilité au risque de taux (la *Modified Duration*) pour chaque obligation, nous voulons savoir si ce spread est "normal" ou "justifié" pour le niveau de risque de l'obligation.
Typiquement, plus la duration est grande (donc plus l'échéance et le risque sont éloignés), plus les investisseurs exigent un spread élevé. Nous modélisons cette relation empiriquement au sein du marché de notre portefeuille.

### La Formule
Nous appliquons une **régression linéaire simple** polynomiale de degré 1 (via `numpy.polyfit`) pour tracer une ligne de tendance optimale entre la *Duration* (en axe X) et le *Z-Spread* (en axe Y).

$$ \text{Fitted Spread} = (\beta_1 \times \text{Duration}) + \beta_0 $$

*où :*
* $\text{Fitted Spread}$ : Le spread "théorique" ou juste valeur de l'obligation tel que dicté par le modèle.
* $\beta_1, \beta_0$ : Les coefficients de la régression linéaire trouvés en minimisant l'erreur.

*Note d'implémentation :* Le paramètre `by_rating_bucket` permet d'exécuter cette régression non pas sur tout le portefeuille de manière globale, mais par sous-catégorie (par exemple, un modèle pour les souverains "Sovereign" et un modèle pour le privé "Corporate"). Cela rend l'analyse bien plus précise car on ne compare entre elles que des obligations appartenant aux mêmes univers de risque de crédit.

## 3. Le screening final : Rich ou Cheap ?

### Le rationnel
Pour isoler les anomalies de prix ou les opportunités, nous examinons la distance (l'écart) entre le spread réellement observé sur le marché, et le spread censé s'appliquer d'après la ligne de tendance. Cet écart s'appelle le **Résidu**.

### La Formule du Résidu
$$ \text{Résidu (bps)} = \text{Z-Spread observé} - \text{Fitted Spread} $$

### Interprétation et Signal (Fonction `_assign_signal`)
L'algorithme distribue des signaux d'investissement clairs en fonction de la valeur de ce résidu :

- **🟢 Signal "CHEAP" (Bon marché)** : `Résidu > +15 bps`
  L'obligation offre un spread *plus élevé* que ce que recommande la ligne de tendance pour sa duration. Étant donné que le rendement est supérieur sans prime de risque additionnelle théorique, le prix de l'obligation est historiquement bas; elle est perçue comme "Cheap", et donc représente potentiellement une opportunité d’achat.

- **🔴 Signal "RICH" (Cher)** : `Résidu < -15 bps`
  L'obligation affiche un spread *inférieur* à la ligne de tendance. Son taux de rendement est trop faible par rapport à ses consœurs de profil similaire. Son prix est donc élevé : la valeur est dite "Rich" (surévaluée).

- **⚪ Signal "FAIR" (Juste Valeur)** : `Entre -15 bps et +15 bps`
  L'écart est minime ou en ligne avec les variations du marché. La valeur est considérée comme correctement évaluée (fair).

---

### Résumé du Workflow du module
1. On déduit mathématiquement le **Z-Spread** de chaque actif à partir de son prix de marché réel.
2. Pour chaque groupe homogène d'actifs, on trace la régression linaire **Duration vs Z-Spread** (le **Fitted Spread**).
3. On calcule l'écart par rapport à cette droite de régression (**Résidu**).
4. Le screening trie et isole les titres : Au-dessus de la droite avec un large écart = **CHEAP** (opportunité), En-dessous = **RICH** (éviter/vendre).

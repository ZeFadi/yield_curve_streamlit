# Comprendre l'Analyse Portajus/Glissement (Carry & Roll-Down)

Ce document explique les concepts et la logique mathématique derrière le module d'analyse *Carry & Roll-Down* (le portage et le glissement le long de la courbe) implémenté dans notre projet (fichier `carry_rolldown.py`). C'est l'un des outils d'attribution de performance les plus importants pour un gérant de portefeuille obligataire (Fixed Income).

## 1. L'Objectif de cette Analyse

Quand un investisseur achète une obligation et la conserve pendant une certaine période (l'horizon, ex: 3 mois), son rendement futur proviendra de plusieurs sources (les coupons, la variation de prix, etc.). 
L'analyse *Carry & Roll-Down* cherche à répondre à la question suivante : **"Combien vais-je gagner sur cet horizon donné _si rien ne change sur le marché_ (si la courbe des taux reste parfaitement immobile) ?"**

Ce rendement attendu se décompose en deux grands moteurs : le **Carry** (Portage) et le **Roll-Down** (Glissement sur la courbe).

---

## 2. Le Carry (Portage Net)

### Qu'est-ce que c'est ?
Le *Carry* représente les revenus "cash" générés simplement par la détention de l'obligation, nets du coût de financement (car souvent les gérants empruntent à court terme pour acheter ces obligations). Ce montant dépend des intérêts versés et de l'évolution des intérêts courus, sans être affecté par les mouvements de prix pur.

### La Formule (dans `compute_carry_rolldown`)

$$ \text{Carry} = \text{Coupons reçus} + (AI_{t1} - AI_{t0}) - \text{Coût de Financement} $$

*où :*
* **$AI_{t1} - AI_{t0}$** : La différence d'intérêts courus (Accrued Interest) entre la date de fin (horizon $t1$) et la date d'achat ($t0$).
* **Coupons reçus** : Les paiements d'intérêts tombés en espèces pendant la période.
  *(Note : `Coupons + (AI_t1 - AI_t0)` représente les revenus de flux nets générés).*
* **Coût de Financement (Funding Cost)** : Ce que coûte l'emprunt de liquidités pour acheter l'obligation (Prix total $\times$ Taux de financement court terme $\times$ Temps écoulé).

### Interprétation
Si l'obligation offre un taux de rendement supérieur au taux auquel vous vous financez, vous générez un carry "positif". Être long sur ce type d'obligation rapporte mécaniquement de l'argent chaque jour qui passe (on "engrange" le portage).

---

## 3. Le Roll-Down (Glissement le long de la courbe)

### Qu'est-ce que c'est ?
Même si la courbe des taux ne bouge absolument pas sur le graphique, votre obligation, elle, va *vieillir*. Une obligation à 10 ans deviendra dans un an une obligation à 9 ans. 
Généralement, la courbe des taux est ascendante (les taux à 10 ans sont plus hauts que les taux à 9 ans). Donc, en vieillissant, l'obligation "glisse" le long de la courbe vers des taux plus bas. 
Une baisse de taux engendre une hausse de prix : **l'obligation s'apprécie mécaniquement**. C'est cet effet pur de vieillissement à courbe constante qu'on appelle le *Roll-Down*.

### La Formule

Pour mesurer cet effet pur (sans qu'il soit "pollué" par le paiement du coupon de tous les jours), l'algorithme utilise les **prix clean** (Prix propre, hors courus).

$$ \text{Roll-Down} = P_{\text{clean\_horizon\_t1}} - P_{\text{clean\_achat\_t0}} $$

*où :*
* $P_{\text{clean\_achat\_t0}}$ : Le prix théorique clean aujourd'hui.
* $P_{\text{clean\_horizon\_t1}}$ : Le prix théorique clean calculé à la date d'horizon (ex: dans 3 mois), mais avec la courbe des taux d'aujourd'hui en estimant que l'échéance de l'obligation s'est rapprochée.

### Interprétation
Si la courbe est "pentue" (steep), l'effet *Roll-Down* est généralement positif et très profitable. Si la courbe est inversée (taux courts plus hauts que taux longs), le *Roll-Down* devient négatif et vient pénaliser la performance.

---

## 4. Total Return (Rendement Total) et Annualisation

### Rendement Total Anticipé à Courbe Constante
Le rendement total (Total Return) prévu sur l'horizon, en l'absence de chocs de marché, est tout simplement la somme des deux mécaniques :

$$ \text{Total Return} = \text{Carry} + \text{Roll-Down} $$

### Annualisation (bps)
Pour que le gérant puisse comparer des investissements sur différents horizons (ex: un investissement à 3 mois vs 6 mois), le code ramène ces gains absolus financiers de l'horizon de temps dans des mesures annualisées en points de base (bps) rapportées à l'investissement initial :

$$ \text{Carry Annualisé (bps)} = \left( \frac{\text{Carry}}{PV_{\text{now}} \times \text{Années d'horizon}} \right) \times 10000 $$

Le module effectue ce même calcul d'annualisation pour le *Roll-Down* et le *Total Return*.

---

### Résumé de l'Impact Métier
L'analyse *Carry & Roll-Down* est souvent considérée comme **l'estimation de la rentabilité "sûre et mécanique" du portefeuille**.
Un Portfolio Manager l'utilise pour :
1. Estimer combien son portefeuille génèrera dans un scénario "base-case" de stabilité (le *Yield-to-Horizon*).
2. Construire des stratégies de positionnement sur la courbe. Au lieu d'acheter simplement un point où le rendement est le plus haut, il cherchera le segment de la courbe qui cumule un bon rendement *ET* la pente la plus forte pour maximiser le "Roll-Down".

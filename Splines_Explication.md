# L'Interpolation par Splines et le Modèle Nelson-Siegel

Ce document explique comment nous construisons la courbe des taux (Yield Curve) de manière continue à partir de points de marché discrets, l'utilité des Splines dans ce contexte, et si/comment le modèle de Nelson-Siegel s'inscrit dans notre programme. La logique centrale se trouve dans le fichier `yield_curve_analyzer.py`.

## 1. Pourquoi avons-nous besoin d'interpolation (Splines) ?

Sur le marché financier, nous n'avons accès qu'à un nombre limité de points de données observables. Par exemple, nous connaissons le taux d'intérêt exact pour les maturités 1 an, 2 ans, 5 ans, 10 ans et 30 ans. 
Cependant, pour valoriser un portefeuille, le système a besoin de connaître le taux précis pour *n'importe quelle* échéance : par exemple exactement 4,25 ans (soit 4 ans et 3 mois).

Pour deviner les taux cachés "entre" les points de marché officiels sans créer de cassures dans le modèle, nous utilisons des techniques mathématiques d'interpolation appelées **Splines**.

## 2. L'Interpolation par Splines (Usage dans le Projet)

Une Spline est une fonction mathématique définie par morceaux à l'aide de polynômes. Dans notre projet, nous utilisons les puissants algorithmes de la bibliothèque `scipy` pour proposer **deux méthodes distinctes d'interpolation par Splines**, chacune répondant à un besoin précis de la gestion de portefeuille :

### A. Natural Cubic Spline (Spline Cubique Naturelle)
* **Qu'est-ce que c'est ?** L'algorithme relie chaque point de marché par un polynôme de degré 3, en s'assurant que la courbe soit parfaitement "lisse" (continuité des dérivées premières et secondes : C2).
* **Avantage** : C'est la courbe visuellement la plus belle et la plus "naturelle".
* **L'inconvénient (Usage métier)** : C'est une interpolation "Globale". Cela signifie qu'un choc sur le taux à 10 ans va légèrement faire bouger le taux interpolé à 2 ans. Cela peut créer des comportements indésirables lorsqu'on calcule les taux *Forward* (Taux à terme), en créant artificiellement des oscillations que le marché ne justifie pas.

### B. PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
* **Qu'est-ce que cest ?** C'est une autre forme de Spline cubique qui est dite "shape-preserving" (qui préserve la forme). 
* **Avantage (Usage métier)** : C'est une interpolation "Locale". Un choc sur le 10 ans n'affectera que la zone entre 7 ans et 15 ans, sans déformer le reste de la courbe. De plus, elle ne crée jamais de "bosses" artificielles (oscillations) entre deux points de marché.
* **Le choix du projet** : C'est la méthode PCHIP qui est paramétrée **par défaut** partout dans nos outils de pricing (`app.py`, `pricing.py`, etc.). Les Portfolio Managers préfèrent grandement le PCHIP car cela rend la couverture de risque (Hedging) prévisible et stable, sans sensibilité "fantôme" (spurious risk) à l'autre bout de la courbe.

---

## 3. Utilisons-nous le modèle de Nelson-Siegel ?

### Verdict
**Non. Le modèle de Nelson-Siegel (ou son extension Nelson-Siegel-Svensson) n'est pas implémenté dans notre projet.**

### Pourquoi ce choix technique ?
Il existe deux grandes écoles pour construire une courbe des taux en finance :

1. **L'approche Paramétrique (Ex: Nelson-Siegel)**
   * *Principe* : On force la courbe à obéir à une seule grande fonction mathématique mondiale pilotée par 3 ou 4 paramètres (qui représentent généralement le Level, la Slope et la Curvature).
   * *Avantage* : Très robuste, lisse, excellente pour la macro-économie et les banques centrales.
   * *Inconvénient* : Le modèle **ne passe pas exactement** par les points de marché observés. Si le taux à 5 ans cote à 3,10% sur le marché, le modèle NS va peut-être l'estimer à 3,12% car il essaie de lisser l'ensemble de la courbe globale. 

2. **L'approche par Interpolation exacte (Ex: Nos Splines & PCHIP)**
   * *Principe* : On relie mathématiquement les points.
   * *Avantage* : **On respecte le marché à 100%**. Si le taux à 5 ans est coté à 3,10%, la spline passera exactement par 3,10%. 
   
**Pour un outil de gestion d'actifs (Asset Management) et de valorisation de portefeuille (Relative Value, Carry, Pricing exact), il est primordial que la courbe construite reprenne exactement les prix de marché réels.** Si notre courbe de référence ignorait le vrai prix du marché pour lisser les données (comme le ferait Nelson-Siegel), toutes nos analyses de "Rich/Cheap" détecteraient de fausses anomalies purement dues à l'erreur du modèle paramétrique.

C'est pour cela que la finance quantitative orientée *trading* et *gestion de fonds* privilégie systématiquement l'interpolation locale exacte (les Splines, et en particulier le PCHIP) par rapport aux modèles paramétriques lissants comme Nelson-Siegel.

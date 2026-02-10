# Yield Curve & Portfolio Manager

**Application professionnelle d'analyse de courbes de taux et de gestion de portefeuille obligataire**

> Outil quantitatif pour la construction de courbes de taux, le pricing de portefeuilles obligataires et les stress tests déterministes.

---

## Fondements Mathématiques

### 1. Courbe des Taux Zéro-Coupon

Le taux zéro-coupon $R(t)$ représente le rendement actuariel continu d'une obligation zéro-coupon de maturité $t$. Le facteur d'actualisation associé est :

$$P(t) = e^{-R(t) \cdot t}$$

### 2. Taux Forward Instantané

Le taux forward instantané $f(t)$ est défini comme la dérivée logarithmique du facteur d'actualisation :

$$f(t) = -\frac{d}{dt}\ln P(t) = \frac{d}{dt}\left[R(t) \cdot t\right]$$

En développant :

$$\boxed{f(t) = R(t) + t \cdot R'(t)}$$

où $R'(t)$ est la dérivée première du taux zéro.

### 3. Taux Forward à Terme

Le taux forward entre deux maturités $t_1$ et $t_2$ (avec $t_1 < t_2$) est :

$$F(t_1, t_2) = \frac{R(t_2) \cdot t_2 - R(t_1) \cdot t_1}{t_2 - t_1}$$

**Exemple :** Le taux 5Y5Y (5 ans dans 5 ans) représente le taux forward à 5 ans observé dans 5 ans :
$$F(5, 10) = \frac{R(10) \times 10 - R(5) \times 5}{10 - 5}$$

---

## Méthodes d'Interpolation

### Spline Cubique Naturelle (Continuité C²)

La spline cubique naturelle assure une courbe deux fois continûment dérivable. Sur chaque segment $[t_i, t_{i+1}]$, le polynôme $S_i(t)$ satisfait :

$$S_i(t) = a_i + b_i(t-t_i) + c_i(t-t_i)^2 + d_i(t-t_i)^3$$

**Conditions aux bords (naturelles) :**
$$S''(t_0) = 0 \quad \text{et} \quad S''(t_n) = 0$$

| Avantages | Inconvénients |
|-----------|---------------|
| Continuité C² (courbe lisse) | Sensibilité globale (propagation des chocs) |
| Dérivées analytiques | Oscillations parasites possibles |
| Adapté au pricing dérivés | Risque de distorsion des forwards |

### PCHIP - Shape-Preserving (Continuité C¹)

Le PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) préserve la monotonie des données. Les pentes $d_i$ aux nœuds sont calculées pour garantir :

$$\text{Si } \Delta_i = \frac{y_{i+1} - y_i}{x_{i+1} - x_i} \text{ et } \Delta_{i-1} \text{ ont le même signe, alors } d_i = \frac{2\Delta_{i-1}\Delta_i}{\Delta_{i-1} + \Delta_i}$$

| Avantages | Inconvénients |
|-----------|---------------|
| Préservation de forme | Continuité C¹ seulement |
| Comportement local | Dérivées secondes discontinues |
| Stabilité des forwards | Moins lisse visuellement |

---

## Pricing Obligataire

### Valeur Actuelle (PV)

Pour une obligation à taux fixe avec des flux de coupon $C_j$ aux dates $t_j$ et remboursement du principal $N$ à maturité $T$ :

$$PV = \sum_{j=1}^{n} C_j \cdot e^{-r(t_j) \cdot t_j} + N \cdot e^{-r(T) \cdot T}$$

où $r(t) = \frac{R(t) + s}{100}$ avec $s$ le spread de crédit en pourcentage.

### Intérêts Courus

Les intérêts courus (Accrued Interest) en convention ACT/ACT sont :

$$AI = C \times \frac{\text{Jours depuis dernier coupon}}{\text{Jours dans la période}}$$

**Prix Clean vs Dirty :**
- **Dirty Price** = PV (valeur totale)
- **Clean Price** = PV - Intérêts Courus

---

## Métriques de Risque

### Duration Modifiée

La duration modifiée mesure la sensibilité du prix aux variations de taux :

$$D_{mod} \approx \frac{1}{PV} \times \frac{PV_{\downarrow} - PV_{\uparrow}}{2\Delta r}$$

où $\Delta r = 1$ bp = 0.0001

### Convexité

La convexité capture l'effet de second ordre (courbure) :

$$C = \frac{1}{PV} \times \frac{PV_{\uparrow} + PV_{\downarrow} - 2 \cdot PV}{(\Delta r)^2}$$

### DV01 (Dollar Value of 01)

Variation de valeur pour un choc de 1 point de base sur la courbe entière :

$$DV01 = \frac{PV_{\downarrow 1bp} - PV_{\uparrow 1bp}}{2}$$

### CS01 (Credit Spread 01)

Variation de valeur pour un choc de 1 bp sur le spread de crédit uniquement :

$$CS01 = \frac{PV_{s-1bp} - PV_{s+1bp}}{2}$$

---

## Scénarios de Stress Test

### 1. Choc Parallèle

Déplacement uniforme de la courbe entière de $\Delta$ bp :

$$R'(t) = R(t) + \frac{\Delta}{100}$$

### 2. Choc de Pente (Twist)

Aplatissement ou pentification avec interpolation linéaire entre court et long terme :

$$R'(t) = R(t) + \frac{1}{100}\left[\Delta_{short} + (\Delta_{long} - \Delta_{short}) \cdot \frac{t - t_{min}}{t_{max} - t_{min}}\right]$$

### 3. Choc Key Rate

Chocs ponctuels à des ténors spécifiques (ex: 2Y, 5Y, 10Y) pour analyser l'exposition aux différentes parties de la courbe :

$$R'(t_k) = R(t_k) + \frac{\Delta_k}{100}$$

---

## Architecture du Projet

```
yield_curve_streamlit/
├── app.py                    # Interface Streamlit principale
├── yield_curve_analyzer.py   # Moteur de construction de courbe
├── pricing.py                # Fonctions de pricing et risque
├── portfolio.py              # Gestion des positions
├── scenarios.py              # Scénarios de stress test
├── requirements.txt          # Dépendances Python
├── sample_portfolio.csv      # Exemple de portefeuille
└── tests/                    # Tests unitaires
```

### Modules Principaux

| Module | Responsabilité |
|--------|----------------|
| `YieldCurveAnalyzer` | Construction de courbe, interpolation, calcul des forwards |
| `price_bond()` | Actualisation des flux, calcul PV/dirty/clean |
| `bond_risk_metrics()` | Duration, convexité, DV01, CS01 |
| `apply_*_shock()` | Application des scénarios de stress |

---

## Installation et Lancement

### Prérequis

- Python 3.9+
- pip

### Installation

```bash
# Cloner ou télécharger le projet
cd yield_curve_streamlit

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Lancement

```bash
streamlit run app.py
```

L'application sera disponible sur `http://localhost:8501`

---

## Utilisation

### 1. Chargement des Données de Marché

L'application supporte plusieurs sources de données :
- **Sample OIS/Govt** : Courbes d'exemple pré-chargées
- **US Treasury** : Téléchargement automatique des données journalières du Trésor américain
- **Upload CSV** : Import de fichiers personnalisés (colonnes : `tenor`, `rate`)
- **Database** : Connexion à une base de données via SQLAlchemy

### 2. Analyse de Courbe

- Visualisation des courbes zéro et forward
- Comparaison des méthodes d'interpolation
- Calcul des forwards à terme (1Y1Y, 2Y3Y, 5Y5Y)
- Stress tests avec analyse de propagation

### 3. Gestion de Portefeuille

- Import de positions obligataires
- Calcul automatique des métriques de risque
- Agrégation au niveau portefeuille

### 4. Analyse de Scénarios

- Chocs parallèles, twist, key rate
- Chocs de spread (credit)
- Calcul du P&L par scénario

---

## Format des Données

### Courbe de Taux (CSV)

```csv
tenor,rate
0.25,3.85
0.5,3.90
1,3.95
2,3.75
5,3.45
10,3.65
30,4.15
```

### Portefeuille (CSV)

```csv
id,issuer,type,currency,notional,coupon_rate,coupon_freq,maturity_date,curve_id,spread_bps
UST_5Y,US Treasury,sovereign,USD,5000000,4.25,2,2030-02-15,UST,0
ACME_28,ACME Corp,corporate,USD,3000000,5.75,2,2028-08-15,UST,180
```

| Colonne | Description | Requis |
|---------|-------------|--------|
| `id` | Identifiant unique | ✅ |
| `issuer` | Nom de l'émetteur | ✅ |
| `type` | `sovereign` ou `corporate` | ✅ |
| `currency` | Code devise (USD, EUR, ...) | ✅ |
| `notional` | Montant nominal | ✅ |
| `coupon_rate` | Taux coupon annuel (%) | ✅ |
| `coupon_freq` | Fréquence des coupons (1, 2, 4) | ✅ |
| `maturity_date` | Date de maturité | ✅ |
| `curve_id` | Identifiant de la courbe | ❌ |
| `spread_bps` | Spread de crédit (bp) | ❌ |

---

## Configuration Avancée

### Variables d'Environnement

- `YC_DB_URL` : URL de connexion à la base de données (format SQLAlchemy)

### Streamlit Secrets

Créer un fichier `.streamlit/secrets.toml` :

```toml
YC_DB_URL = "postgresql://user:password@host:port/database"
```

---

## Références Théoriques

1. **Interpolation de courbes** : Hagan, P.S. & West, G. (2006). *Interpolation Methods for Curve Construction*. Applied Mathematical Finance.

2. **Duration et Convexité** : Fabozzi, F.J. (2007). *Fixed Income Analysis*. CFA Institute.

3. **Splines cubiques** : De Boor, C. (1978). *A Practical Guide to Splines*. Springer-Verlag.

4. **PCHIP** : Fritsch, F.N. & Carlson, R.E. (1980). *Monotone Piecewise Cubic Interpolation*. SIAM Journal on Numerical Analysis.

---

## Avertissement

Cette application est fournie à titre éducatif et de démonstration. Les données du Trésor américain importées sont des rendements "par" (par yields) et non des taux zéro - une étape de bootstrap serait nécessaire pour une utilisation en production.

Pour toute décision d'investissement, consultez un professionnel qualifié.

---

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

*Développé pour la finance quantitative*


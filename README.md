# Streamlit x_min / x_max / k diagnostics

Application Streamlit dediee au calibrage de la formule de score y(x) avec:

- un choix d indicateur,
- des bornes x_min / x_max,
- un parametre de courbure k > 0.

## Fonctionnalites

- Selection d un indicateur iXXX depuis valeur_externe.csv.
- Visualisations d aide a la decision:
  - histogramme avec bornes candidates,
  - boxplot,
  - CDF,
  - courbes y(x) pour plusieurs valeurs de k,
  - diagnostic de k (equilibre, moyenne, dispersion, extremes),
  - distribution des scores calcules.
- Controle interactif de:
  - la direction (croissant ou decroissant),
  - x_min / x_max,
  - k applique.
- Exports CSV:
  - resume global par indicateur,
  - donnees avec scores calcules.

## Donnees attendues

Par defaut, l app lit le fichier local:

- `valeur_externe.csv`

Le chemin est modifiable dans la barre laterale.

## Lancement

```bash
cd /home/hadrien/Documents/projet_code/streamlit_k_repo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploiement Streamlit Cloud

- Le fichier `runtime.txt` force Python 3.12 pour eviter des differences de runtime.
- Le fichier `.streamlit/config.toml` desactive `gatherUsageStats` afin de limiter les appels telemetry tiers (Segment/Heap).

Notes:

- Les erreurs console navigateur sur `segment.io` ou `heapanalytics.com` sont en general non bloquantes pour l application.
- En cas de probleme, verifier les logs serveur Streamlit (traceback Python) plutot que les warnings JS telemetry.

## Formule utilisee

- Normalisation croissante:
  - u(x) = clip((x - x_min) / (x_max - x_min), 0, 1)
- Normalisation decroissante:
  - u(x) = clip((x_max - x) / (x_max - x_min), 0, 1)
- Score:
  - y(x) = 100 * (exp(k * u(x)) - 1) / (exp(k) - 1), avec k > 0

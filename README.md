# Streamlit x_min / x_max / k diagnostics

Application Streamlit dediee au calibrage de la formule de score y(x) avec:

- un choix d indicateur,
- des bornes x_min / x_max,
- un parametre de courbure k > 0.

## Fonctionnalites

- Sélection d’un indicateur dans le CSV déposé.
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

Dans la barre latérale, utilisez **Déposer un fichier CSV**, puis choisissez une colonne dans **Indicateur**. Le dépôt d’un CSV est requis pour afficher les diagnostics et les exports.

Le CSV doit avoir une ligne d’en-tête. Les colonnes de type `i012` ou `i165` sont détectées automatiquement. En l’absence de ces codes, l’application propose les colonnes numériques. Les séparateurs virgule et point-virgule sont acceptés, ainsi que les virgules décimales des exports Excel.

## Lancement

```bash
cd /home/hadrien/Documents/projet_code/streamlit_k_repo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploiement Streamlit Cloud

- Le fichier `.python-version` fixe Python 3.12 cote repo (outils bases sur uv/pyenv).
- Le fichier `runtime.txt` est garde pour compatibilite descendante.
- Le fichier `.streamlit/config.toml` desactive `gatherUsageStats` afin de limiter les appels telemetry tiers (Segment/Heap).

Important:

- Sur Streamlit Community Cloud, la version Python active se choisit dans `Advanced settings` de l app (champ `Python version`).
- Si une app existante reste sur une ancienne version, ouvrir `Manage app` -> `Settings` -> `Python version`, choisir 3.12, sauvegarder, puis `Reboot app`.

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

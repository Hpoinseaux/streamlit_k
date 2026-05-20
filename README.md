# Streamlit k diagnostics

Application Streamlit dediee au choix des bornes et du parametre k par indicateur.

## Fonctionnalites

- Selection d un indicateur iXXX dans une liste.
- Visualisation des 4 graphes de diagnostic:
  - histogramme brut + bornes candidates,
  - CDF brute,
  - comparaison des courbes de scoring selon k,
  - score d equilibre pour chaque k.
- Affichage detaille:
  - top k pour l indicateur,
  - table des bornes candidates,
  - distribution du score avec k recommande,
  - tableau global exportable en CSV.

## Donnees attendues

Par defaut l app lit:

- `../valeur_externe.csv`
- `../streamlit_diag/source/Parametres_indicateurs.csv`

Vous pouvez modifier ces chemins dans la barre laterale.

## Lancement

```bash
cd streamlit_k
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Bornes manuelles (optionnel)

Dans la barre laterale, champ "Bornes manuelles", format:

```text
i005: 5, 85
i071: 0, 120
```

Une ligne par indicateur. Ces bornes remplacent le choix automatique pour les indicateurs renseignes.

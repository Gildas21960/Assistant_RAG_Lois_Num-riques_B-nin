# Assistant RAG — Lois numériques du Bénin
**AI4Youth / Ed. Lomé 2026 — Catégorie : Application (RAG / LLM intégré)**

Chatbot qui répond, en langage simple, aux questions sur le droit du numérique au Bénin (ex. : « quelqu'un a publié ma photo sans mon accord, que dit la loi ? »). Il s'appuie **uniquement** sur les textes de loi indexés, **cite les articles** et **refuse de répondre** quand aucun extrait n'est assez pertinent.

> ⚠️ Information générale, qui ne remplace pas l'avis d'un avocat ou d'un professionnel du droit.

## Contenu du dépôt
| Fichier | Rôle |
|---|---|
| `Assistant_RAG_Lois_Numeriques_Benin.ipynb` | Livrable principal : pipeline complet, EDA, évaluation, analyse |
| `rag_core.py` | Pipeline RAG (généré depuis le notebook, réutilisé par l'app et le CLI) |
| `app.py` | Application web Streamlit |
| `eval_questions.json` | Jeu de test annoté (questions + articles attendus) |
| `corpus/` | Textes de loi (PDF) — voir « Sources » |
| `requirements.txt` | Dépendances |

## Architecture
PDF → nettoyage → **chunking par article** → embeddings `BAAI/bge-m3` → FAISS + BM25 → (multi-query) → recherche hybride (fusion RRF) → re-ranking cross-encoder → seuil de pertinence → prompt augmenté → Gemini → réponse + articles cités.
Détails, choix et justifications : voir le notebook (sections 5 et 9).

## Installation
```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="votre_clé"                       # Windows : set GEMINI_API_KEY=votre_clé
```
Ne committez **jamais** votre clé API.

## Reproduire les résultats
1. Placez les PDF de lois dans `corpus/` (le nom de fichier devient le nom de la loi dans les citations).
2. Renseignez le champ `expected` de `eval_questions.json` (format `nom_fichier::numero_article`).
3. Ouvrez le notebook (Colab ou Jupyter) → **Exécuter tout** : il construit l'index (`index/`), évalue les configurations et exporte `generation_review.csv` (à annoter manuellement).
   - Sur Colab : ajoutez `GEMINI_API_KEY` dans les *Secrets*, activez si possible un GPU pour accélérer l'indexation.

## Quota API et cache
L'API Gemini gratuite est limitée (quota par jour et par modèle). Le code met en cache les réponses LLM dans `llm_cache/`, distingue les erreurs de quota (`QuotaExceeded`) des erreurs temporaires, et n'appelle pas le LLM pour les tests de retrieval et de refus. Pour un usage intensif, utilisez une clé avec facturation ou les crédits fournis par l'organisation.

## Lancer l'application
```bash
streamlit run app.py                     # interface web
python rag_core.py "Ma question ici"     # ligne de commande
```
(Nécessite le dossier `index/` créé par le notebook.)

## Évaluation (résumé)
Les résultats chiffrés (Hit@k, MRR, refus, fidélité, latence, coût) sont produits par le notebook et détaillés en section 10. *Reportez ici les chiffres finaux obtenus.*

## Limites
Corpus limité aux textes fournis (peut être incomplet ou dépassé) ; jeu de test de petite taille ; le LLM peut mal interpréter un article ; extraction PDF parfois imparfaite. Voir notebook section 11.

## Sources et licences
- Loi n° 2017-20 du 20 avril 2018 portant Code du numérique en République du Bénin (PDF public ; attention : la copie de l'APDP a une extraction de texte corrompue, que le pipeline répare par heuristique — préférez une copie propre).
- Loi n° 2018-16 portant Code pénal (PDF public, Assemblée nationale).
- *Non incluse (limite connue) :* loi n° 2020-35 modifiant le Code du numérique.
- *À compléter :* liens exacts de téléchargement utilisés, modèles tiers (bge-m3, bge-reranker-v2-m3, cross-encoder mmarcoFR, Gemini) et leurs licences.

## Équipe
*GNONHOSSOU Gildas Mahugnon*
*QUENUM Mahulahen Eustache Ornel*
*BIAHOU Ahossou Junior Marie-Clémenceau* 
*ADOGOUN Déo Gracias Mahugnon*

# Prompt Codex — POC `Laya Dynamics Agent`

Tu es **GPT-5.6 Sol dans Codex**, exécuté depuis **VSCodium sous Debian 13 Trixie XFCE**.

Ta mission est de concevoir et implémenter un **proof of concept de recherche reproductible** qui teste une hypothèse précise :

> **Un modèle Laya local post-entraîné sur les transitions réelles d'un agent peut-il apprendre des propriétés utiles de la dynamique d'une tâche et améliorer la planification d'un LLM génératif par rapport à GPT seul ?**

Ce projet n'a PAS pour objectif de prétendre construire une AGI.

Nous cherchons à tester une brique qui pourrait être utile dans une architecture plus générale :

```text
LLM génératif
-> actions candidates
-> petit modèle local spécialisé
-> prédiction des conséquences / valeur
-> sélection
-> action réelle
-> observation réelle
-> erreur de prédiction
-> nouvelles données d'entraînement
```

---

## 0. Contexte utilisateur

J'ai déjà construit un agent contrôlant une boutique Shopify avec GPT.

J'ai également expérimenté auparavant l'ajout de TypeSafe/JEV comme couche de décision.

Les résultats ont été mitigés :

- complexité supplémentaire ;
- bénéfice pas toujours évident ;
- possible dégradation de certaines décisions ;
- dépendance à un service externe ;
- latence/coût supplémentaires ;
- comportement trop générique pour certains états métier.

Cette expérience sert uniquement de **retour d'expérience architectural** : **JEV est hors périmètre de ce projet**.

Je ne veux :

- ni fallback JEV ;
- ni baseline JEV ;
- ni dépendance JEV ;
- ni couche de compatibilité JEV ;
- ni abstraction ajoutée uniquement pour pouvoir réintroduire JEV plus tard.

Je veux maintenant tester une approche plus rigoureuse avec **Laya en local**, spécialisée par post-entraînement sur les trajectoires réelles de l'agent.

IMPORTANT :

- ne traite pas Laya comme un simple « judge » ;
- ne l'appelle pas systématiquement sans mesurer sa valeur ;
- il doit apprendre des propriétés réellement corrélées aux conséquences observées des actions ;
- toute conclusion doit être démontrée par benchmark.

Le modèle Laya complet fonctionne déjà localement sur la machine cible et tient en VRAM en inférence. Ne construis donc pas le projet autour d'une hypothèse artificielle disant que le checkpoint complet est trop gros.

Tu dois néanmoins mesurer et logger proprement :

```text
VRAM
RAM
latence
débit
temps de chargement
```

---

# 1. Avant de coder : auditer l'état actuel de Laya

Commence par consulter la documentation et le dépôt officiel actuels :

```text
https://github.com/NandhaKishorM/laya
```

En particulier :

```text
README
docs/
docs/finetune_browser_agent.md
notebooks/
examples de fine-tuning
calibration
choice / score / noul
```

Ne suppose pas que l'API actuelle correspond à une ancienne version.

Vérifie :

- version disponible ;
- dépendances ;
- licences ;
- checkpoints ;
- API Python ;
- format des questions ;
- format de fine-tuning ;
- méthode de calibration ;
- limitations de longueur ;
- batching ;
- comportement CUDA ;
- possibilité de hooks ;
- exemples browser-agent.

Documente les résultats dans :

```text
docs/LAYA_AUDIT.md
```

avec les versions exactes testées.

---

# 2. Environnement cible

OS :

```text
Debian 13 Trixie
XFCE
Linux x86_64
```

IDE :

```text
VSCodium
```

GPU utilisateur principal :

```text
NVIDIA RTX 3070 Ti
8 GB VRAM
```

RAM :

```text
32 GB DDR4
```

Utilise Python moderne compatible avec Debian 13 et Laya.

Préférence :

```text
uv
```

si approprié.

Sinon :

```text
venv + pip
```

mais explique le choix.

Ne modifie pas globalement le Python système Debian.

Créer un environnement isolé.

---

# 3. Nom du projet

Nom de travail :

```text
laya-dynamics-agent
```

Le nom pourra être changé plus tard.

---

# 4. Hypothèse expérimentale

Formalise dans :

```text
docs/HYPOTHESIS.md
```

## H0

Laya spécialisé n'améliore pas suffisamment les performances de l'agent pour justifier sa complexité.

## H1

Un petit decision model local entraîné sur :

```text
(state_before, action, state_after, outcome)
```

peut estimer certaines propriétés de transition suffisamment bien pour améliorer les décisions du planner GPT.

---

# 5. Architecture obligatoire

Commencer avec une architecture simple.

```text
                   Observation
                        |
                        v
                  StateBuilder
                        |
                       s_t
                        |
                        v
                 GPT Planner
                        |
             candidate actions
                        |
                        v
              Laya Dynamics Head
                        |
         predicted action properties
                        |
                        v
                     Policy
                        |
                        v
                  chosen action
                        |
                        v
                  Environment
                        |
                        v
                      s_t+1
                        |
          +-------------+-------------+
          |                           |
          v                           v
 prediction evaluation          trajectory store
          |                           |
          +-------------+-------------+
                        |
                        v
                  training dataset
```

Tout composant doit être remplaçable.

Interfaces propres.

Pas de spaghetti code.

---

# 6. State schema

Créer un schéma Pydantic versionné.

Exemple de base :

```python
class AgentState(BaseModel):
    schema_version: str

    task_id: str
    step_index: int

    goal: str

    observations: list[Observation]
    beliefs: list[Belief]
    unknowns: list[str]

    history: list[ActionRecord]

    environment: dict

    terminal: bool
```

Mais ne copie pas aveuglément cet exemple.

Conçois quelque chose de :

- compact ;
- sérialisable ;
- stable ;
- déterministe ;
- compatible dataset ;
- compatible migration future.

Chaque état doit recevoir un hash reproductible.

---

# 7. Action schema

Créer une représentation explicite :

```python
class CandidateAction(BaseModel):
    action_id: str
    tool: str
    args: dict
    rationale_short: str | None
```

Les actions ne doivent pas être des chaînes libres impossibles à analyser.

Les tools doivent avoir un JSON schema.

---

# 8. Laya ne doit PAS choisir uniquement « la meilleure action »

Le but est d'estimer les propriétés de chaque transition potentielle.

Commencer avec ces sorties :

```text
P(success)
P(goal_progress)
P(information_gain)
P(risk)
P(reversible)
P(needs_more_observation)
```

Si les primitives Laya imposent `choice`, `score` et `noul`, mappe ces concepts proprement.

Documente exactement comment.

Exemple :

```text
noul:
"Will this action probably advance the current goal?"

score:
"How much useful information is this action expected to reveal?"
0..4

noul:
"Can this action create a hard-to-reverse external side effect?"
```

Ne transforme pas artificiellement toutes les sorties en probabilités si la primitive ne le permet pas correctement.

---

# 9. Policy explicite

La décision finale ne doit pas être cachée dans Laya.

Créer une `Policy`.

Exemple conceptuel :

```python
utility =
    w_success * success
    + w_progress * goal_progress
    + w_information * information_gain
    - w_risk * risk
    + w_reversible * reversible
```

Mais :

- les poids doivent être configurables ;
- ils doivent être loggés ;
- aucune valeur magique non documentée ;
- permettre plusieurs policies ;
- séparer prédiction et sélection.

Ajouter au minimum :

```text
GreedyUtilityPolicy
GPTOnlyPolicy
RandomPolicy
OraclePolicy
```

`OraclePolicy` seulement lorsque le sandbox permet de connaître le meilleur choix.

---

# 10. Environnement expérimental

NE COMMENCE PAS PAR SHOPIFY.

Créer d'abord un environnement web local et contrôlé.

Il doit pouvoir simuler des tâches telles que :

```text
trouver une information
ouvrir la bonne page
identifier une source primaire
comparer plusieurs valeurs
remplir un formulaire
naviguer sur plusieurs étapes
corriger une hypothèse
```

Préférence :

```text
FastAPI ou Flask
+
Playwright
```

ou environnement encore plus simple si suffisant.

Le site sandbox doit pouvoir générer des variantes automatiquement.

Exemples :

```text
/product/A/specs
/product/B/specs
/docs/...
/forum/...
/manufacturer/...
```

Créer parfois :

- sources contradictoires ;
- pages obsolètes ;
- informations manquantes ;
- faux raccourcis ;
- actions irréversibles simulées ;
- formulaires ;
- résultats de recherche.

Les tâches doivent disposer d'un ground truth.

---

# 11. Important : abstraction avant DOM brut

Ne donne pas nécessairement tout le DOM à Laya.

La documentation browser-agent de Laya montre que le format d'entrée et la présentation des candidats influencent fortement les performances.

Construire une représentation compacte :

```text
page title
URL/path
short relevant page text
candidate elements/actions
recent history
goal
known facts
unknowns
```

Mesurer plusieurs formats.

Ne jamais confondre :

```text
meilleur modèle
```

et :

```text
meilleur formatting
```

---

# 12. Planner GPT

Créer une interface générique :

```python
class Planner(Protocol):
    async def propose_actions(
        self,
        state: AgentState,
        max_actions: int
    ) -> list[CandidateAction]:
        ...
```

Implémenter :

```text
OpenAIPlanner
```

mais isoler totalement le provider.

Les prompts doivent être versionnés :

```text
prompts/planner_v001.md
```

Ne mettre aucune clé API dans le dépôt.

Lire les secrets depuis :

```text
.env
```

Ajouter :

```text
.env.example
```

---

# 13. Candidate generation

GPT ne doit pas directement sélectionner une seule action dans le mode expérimental principal.

Il doit proposer par exemple :

```text
3 à 8 actions
```

configurable.

Il faut enregistrer :

```text
candidate set
ordre proposé
arguments
prompt version
model
temperature
token usage
latency
```

---

# 14. Trajectory store

Utiliser SQLite pour commencer.

Stocker au minimum :

```text
run
task
step
state_before
candidate_actions
laya_predictions
chosen_action
state_after
observed_transition
reward/outcome
LLM usage
timings
errors
versions
```

La DB doit permettre de reconstruire entièrement une exécution.

Prévoir migration/version.

---

# 15. Event log

Créer en parallèle un log JSONL append-only.

Exemple :

```text
logs/runs/<run_id>/events.jsonl
```

Chaque événement doit avoir :

```text
timestamp
run_id
task_id
step
event_type
payload
```

Aucun secret.

---

# 16. Construction automatique des labels

C'est une partie centrale.

À partir de :

```text
state_before
action
state_after
ground_truth
```

construire autant que possible automatiquement :

```text
success
goal_progress
information_gain
risk_event
irreversible_side_effect
needs_more_observation
```

Ne demande pas à GPT de créer tous les labels si le sandbox permet de les obtenir de manière déterministe.

Règle :

```text
deterministic label > heuristic label > LLM label
```

Tout label doit enregistrer sa provenance.

---

# 17. Information gain

Ne prétends pas calculer une information théorique parfaite si nous n'avons pas une distribution probabiliste complète.

Pour le POC, créer une approximation clairement documentée.

Exemple :

```text
unknown facts resolved
contradictions resolved
belief confidence improved
ground-truth-relevant variables discovered
```

Prévoir plusieurs définitions testables.

---

# 18. Dataset

Créer :

```text
data/raw/
data/processed/
data/splits/
```

Chaque sample doit relier :

```text
s_t
a_t
s_t+1
labels
metadata
```

Éviter de dupliquer inutilement les gros blobs.

Les splits doivent être définis par tâche/site/template, pas simplement par lignes aléatoires.

Objectif :

```text
train
validation
calibration
test
```

Le jeu de calibration doit être séparé du train.

Le test doit rester réellement unseen.

---

# 19. Fine-tuning Laya

Adapter les outils officiels de Laya plutôt que réinventer son entraînement sans raison.

Reproduire d'abord un mini fine-tune officiel ou proche de l'exemple browser-agent.

Puis spécialiser sur notre dataset.

Conserver :

```text
configs/train/*.yaml
```

Enregistrer :

```text
base checkpoint
git commit
dataset version/hash
hyperparameters
seed
CUDA version
PyTorch version
Laya version
training duration
peak VRAM
```

Le checkpoint produit doit être identifiable.

---

# 20. Calibration

Indispensable.

Évaluer :

```text
Brier score
log loss
ECE
reliability bins
accuracy / balanced accuracy
```

si la sortie s'y prête.

Ne jamais déclarer :

```text
0.91 = 91 % de probabilité réelle
```

sans calibration évaluée.

Créer des reliability diagrams.

---

# 21. Baselines obligatoires

## Baseline A — GPT direct

```text
GPT reçoit state
-> choisit action
```

## Baseline B — GPT candidate generation + heuristic policy

Pas de Laya.

Cela permet de séparer :

```text
gain du multi-candidate planning
```

de :

```text
gain de Laya
```

## Baseline C — GPT + Laya base zero-shot

## Baseline D — GPT + Laya fine-tuné

## Baseline E — oracle sandbox

Pour connaître le plafond théorique quand possible.


---

# 22. Ablations indispensables

Tester au moins :

```text
sans history
sans beliefs
sans information_gain
sans risk
sans Laya
Laya zero-shot
Laya fine-tuned
1 candidate
3 candidates
5 candidates
8 candidates
```

Si possible également :

```text
texte brut vs état structuré
```

---

# 23. Métriques finales

Créer un benchmark runner.

Pour chaque configuration :

```text
task success rate
mean steps to success
median steps
unnecessary actions
loops detected
tool failures
unsafe/irreversible simulated actions
LLM input tokens
LLM output tokens
estimated API cost
Laya latency
GPT latency
total latency
GPU peak memory
RAM peak
```

Pour Laya :

```text
accuracy
top-k if applicable
Brier
log loss
ECE
calibration plot
```

---

# 24. Statistiques

Ne compare pas deux configs sur 5 tâches et ne conclus pas.

Utiliser suffisamment de runs.

Fixer et varier les seeds.

Produire :

```text
mean
median
std
confidence intervals
```

Utiliser bootstrap si adapté.

Conserver les raw results.

---

# 25. Critère de succès

Définir AVANT les expériences une première règle.

Par exemple :

Laya spécialisé est intéressant si :

```text
TaskSuccess >= GPT-only
AND
mean unnecessary actions decreases
AND
mean API token usage does not materially increase
AND
added decision latency remains small relative to GPT latency
AND
fine-tuned Laya clearly beats zero-shot Laya
```

Ne manipule pas les critères après avoir vu les résultats.

Documenter les critères dans :

```text
docs/EVALUATION_PROTOCOL.md
```

---

# 26. DAgger / on-policy corrections

Seulement après le premier fine-tuning.

Workflow :

```text
run specialised Laya
-> collect failure states
-> obtain correct labels from sandbox/oracle
-> append correction dataset
-> retrain
-> evaluate on untouched test set
```

Ne jamais entraîner directement sur le test set.

---

# 27. Prediction error

Créer un module central.

Pour chaque action exécutée :

```text
predicted_transition_properties
vs
observed_transition_properties
```

Exemple :

```json
{
  "predicted": {
    "goal_progress": 0.82,
    "risk": 0.05
  },
  "observed": {
    "goal_progress": 0.40,
    "risk_event": false
  }
}
```

Calculer et logger l'erreur.

Nous voulons pouvoir tracer :

```text
prediction quality vs training iteration
```

---

# 28. Phase ultérieure : latent dynamics

NE PAS implémenter dans la première milestone.

Créer seulement :

```text
docs/FUTURE_LATENT_DYNAMICS.md
```

Décrire comment on pourrait passer de :

```text
f(s,a) -> typed transition properties
```

à :

```text
f(z_s,z_a) -> delta_z
```

puis :

```text
z_{t+1_hat} = z_t + delta_z
```

afin de permettre un jour des rollouts internes multi-step.

Mais ne détourne pas le POC vers cette direction avant validation du one-step predictor.

---

# 29. Sécurité expérimentale

Le premier environnement doit être sandboxé.

Aucune action :

```text
achat réel
publication Shopify réelle
suppression de données réelle
message externe réel
commande destructive système
```

Le planner doit avoir une allowlist de tools.

Les actions modifiant l'état doivent être simulées ou réversibles.

---

# 30. Détection de boucles

Implémenter :

```text
same state hash repeated
same action repeated
oscillation A-B-A-B
no-progress counter
max steps
```

Logguer les causes d'arrêt.

---

# 31. Bypass de Laya

Très important.

L'architecture doit permettre :

```text
USE_LAYA=false
```

sans changer le reste de l'agent.

Nous voulons savoir si Laya améliore le système.

Il ne doit jamais devenir une dépendance structurelle impossible à retirer.

---

# 32. Tests

Utiliser `pytest`.

Minimum :

```text
unit tests
schema tests
state hashing tests
trajectory reconstruction tests
policy tests
label generation tests
dataset split leakage tests
Laya adapter tests
planner mock tests
sandbox task tests
benchmark smoke test
```

Les tests API doivent pouvoir utiliser des mocks.

---

# 33. Reproductibilité

Créer :

```text
scripts/bootstrap.sh
scripts/run_demo.sh
scripts/run_benchmark.sh
scripts/build_dataset.sh
scripts/train_laya.sh
scripts/evaluate_laya.sh
```

Aucune commande magique uniquement connue de l'auteur.

---

# 34. CLI

Créer une CLI ergonomique.

Exemples souhaités :

```bash
uv run lda doctor
uv run lda demo
uv run lda collect --tasks 100
uv run lda dataset build
uv run lda laya benchmark
uv run lda train --config configs/train/v001.yaml
uv run lda eval --checkpoint checkpoints/v001
uv run lda benchmark --suite heldout
uv run lda report latest
```

Les noms exacts peuvent être ajustés.

---

# 35. `doctor`

La commande :

```bash
lda doctor
```

doit vérifier :

```text
Python
CUDA
GPU
VRAM
PyTorch CUDA
Laya import
checkpoint loading
SQLite
Playwright
browser
OpenAI configuration
disk space
```

Afficher des diagnostics clairs.

---

# 36. Monitoring

Pendant les benchmarks, enregistrer :

```text
GPU utilisation
VRAM
RAM
CPU
latencies
```

Pas besoin d'une grosse stack Prometheus.

CSV/JSONL suffit au début.

---

# 37. Reports

Générer automatiquement :

```text
reports/<experiment_id>/
```

avec :

```text
summary.md
metrics.json
runs.csv
failures.md
calibration.png
latency.png
task_success.png
steps.png
cost.png
```

Ne jamais utiliser uniquement des moyennes.

Inclure quelques trajectoires représentatives.

---

# 38. README

Le README doit expliquer :

1. la question de recherche ;
2. ce que Laya fait réellement ;
3. ce que le projet ne prétend pas faire ;
4. l'architecture ;
5. installation Debian 13 ;
6. configuration OpenAI ;
7. lancer le sandbox ;
8. lancer une tâche ;
9. collecter les trajectoires ;
10. entraîner Laya ;
11. exécuter les benchmarks ;
12. lire les rapports.

Ajouter un schéma Mermaid.

---

# 39. Repo structure suggérée

Tu peux améliorer cette structure :

```text
laya-dynamics-agent/
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── configs/
│   ├── agent/
│   ├── benchmark/
│   └── train/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── HYPOTHESIS.md
│   ├── LAYA_AUDIT.md
│   ├── EVALUATION_PROTOCOL.md
│   └── FUTURE_LATENT_DYNAMICS.md
├── prompts/
├── src/
│   └── laya_dynamics_agent/
│       ├── cli/
│       ├── core/
│       ├── planner/
│       ├── laya_adapter/
│       ├── policy/
│       ├── environment/
│       ├── storage/
│       ├── labeling/
│       ├── dataset/
│       ├── training/
│       ├── evaluation/
│       └── telemetry/
├── sandbox/
├── scripts/
├── tests/
├── data/
├── logs/
├── checkpoints/
└── reports/
```

---

# 40. Git discipline

Travaille par commits petits et cohérents.

Exemples :

```text
chore: bootstrap project and environment diagnostics
feat: add versioned state and action schemas
feat: add deterministic sandbox environment
feat: add trajectory event store
feat: add OpenAI candidate planner
feat: add Laya prediction adapter
feat: add transition labeler
feat: add evaluation baselines
feat: add Laya training pipeline
feat: add calibration report
```

Ne mets jamais :

```text
API keys
.env
checkpoints énormes
datasets générés massifs
logs privés
```

dans Git.

---

# 41. Orchestration Sol + sous-agents Luna

Le modèle principal est **GPT-5.6 Sol**.

Sol est l'orchestrateur et reste responsable de :

- l'architecture globale ;
- les interfaces entre composants ;
- les choix expérimentaux ;
- la définition des métriques ;
- les changements touchant plusieurs sous-systèmes ;
- la résolution des conflits ;
- la validation finale ;
- les décisions pouvant affecter la reproductibilité scientifique ;
- les changements à risque ou difficiles à annuler.

Lorsque l'environnement Codex permet d'utiliser des **sous-agents GPT-5.6 Luna**, utilise-les activement pour les tâches qui sont :

```text
bien délimitées
parallélisables
faiblement couplées
faciles à vérifier
```

L'objectif est de réduire le temps passé par Sol sur les tâches mécaniques ou exploratoires sans dégrader la cohérence du projet.

## Tâches à déléguer prioritairement à Luna

Exemples adaptés :

```text
audit ciblé d'un dossier du repo Laya
résumé d'une partie précise de documentation
inventaire d'API ou de configs
recherche de tests existants
écriture de tests unitaires pour un module déjà spécifié
création de fixtures
génération de cas de sandbox
analyse de logs
comparaison de fichiers de benchmark
production de tableaux de métriques
vérification statique d'un module
relecture de documentation
mise à jour d'un README après modification déjà décidée
petits refactors locaux à interface inchangée
implémentation d'un adaptateur isolé dont l'interface est déjà définie
```

Utilise plusieurs Luna en parallèle lorsque les tâches sont indépendantes.

Exemple :

```text
Sol
├── Luna A -> audit fine-tuning Laya
├── Luna B -> inspecte calibration et typed decisions
├── Luna C -> inspecte browser-agent example
└── Sol    -> synthétise et décide l'architecture
```

ou :

```text
Sol
├── Luna A -> tests StateBuilder
├── Luna B -> fixtures sandbox
├── Luna C -> instrumentation mémoire/latence
└── Sol    -> intègre, exécute la suite et corrige
```

## Tâches que Luna ne doit pas décider seul

Ne délègue pas entièrement à Luna :

```text
architecture générale
choix du protocole expérimental
définition de H0/H1
définition des baselines
choix de split train/test
modification simultanée de nombreux modules fortement couplés
migration de schéma destructive
sécurité de l'environnement
policy finale
critères de réussite
interprétation finale des résultats
```

Luna peut analyser ces sujets et proposer des options, mais **Sol prend la décision finale**.

## Contrat de délégation

Pour chaque sous-tâche Luna :

1. donner un objectif précis ;
2. indiquer les fichiers autorisés ou la zone à inspecter ;
3. fournir les interfaces déjà décidées ;
4. demander un résultat vérifiable ;
5. éviter deux sous-agents modifiant le même fichier en parallèle ;
6. faire relire/intégrer le résultat par Sol ;
7. exécuter les tests concernés après intégration.

Ne fusionne jamais mécaniquement une proposition Luna parce qu'elle semble plausible.

Sol doit vérifier :

```text
code
tests
interfaces
hypothèses
effets de bord
```

## Gestion du contexte

Ne donne pas à chaque Luna l'intégralité du projet si ce n'est pas nécessaire.

Préférer :

```text
contexte minimal
+ tâche précise
+ critères d'acceptation
```

afin de réduire :

- bruit ;
- divergence ;
- consommation de contexte ;
- modifications inutiles.

## Fallback si les sous-agents ne sont pas disponibles

Si l'environnement Codex actuel ne fournit pas de mécanisme de sous-agents, **ne bloque pas le projet** et ne simule pas artificiellement plusieurs agents.

Sol poursuit directement les tâches.

Documente simplement dans le journal de travail :

```text
Luna subagents unavailable in current environment
```

Aucun autre modèle ou service externe ne doit être introduit pour remplacer Luna.

---

# 42. Mode de travail Codex

Tu dois :

1. inspecter avant de modifier ;
2. expliquer brièvement les choix d'architecture importants ;
3. exécuter les tests après les changements ;
4. ne jamais prétendre qu'un test passe sans l'avoir exécuté ;
5. ne pas inventer une API Laya ;
6. consulter sa documentation actuelle si nécessaire ;
7. ne pas installer globalement des dépendances système sans nécessité ;
8. demander le minimum d'interventions manuelles ;
9. continuer de manière autonome lorsque l'information est disponible localement ;
10. documenter les limitations réelles.

---

# 43. Première milestone

La première milestone doit s'arrêter à un POC end-to-end minimal :

```text
sandbox task
-> state
-> GPT proposes N candidate actions
-> Laya scores each candidate
-> policy picks one
-> sandbox executes
-> next state produced
-> prediction vs reality logged
-> trajectory stored
```

PLUS :

```text
GPT-only baseline
```

et :

```text
benchmark on a small deterministic suite
```

Ne commence pas le fine-tuning avant que cette boucle soit fiable.

---

# 44. Deuxième milestone

Une fois la boucle stable :

```text
collect trajectories
-> deterministic labels
-> train/val/calibration/test split
-> fine-tune Laya
-> calibrate
-> evaluate
```

Comparer :

```text
GPT-only
GPT + heuristic
GPT + base Laya
GPT + specialised Laya
```

---

# 45. Troisième milestone

Après résultat positif :

```text
DAgger / on-policy correction
harder sandbox tasks
larger action spaces
cross-template generalisation
```

Seulement ensuite envisager :

```text
Shopify adapter
```

---

# 46. Shopify — plus tard

L'intégration Shopify devra être une couche d'environnement supplémentaire.

Ne mélange pas ses objets métiers avec les abstractions core.

Exemple :

```text
Environment protocol
├── SandboxWebEnvironment
└── ShopifyEnvironment   # future
```

Le core doit rester identique.

---

# 47. Question scientifique à garder en permanence

Pour chaque fonctionnalité, demande :

> Est-ce que cela nous aide à déterminer si Laya apprend réellement quelque chose d'utile sur les conséquences des actions ?

Si la réponse est non, reporte cette fonctionnalité.

---

# 48. Anti-objectifs

Évite absolument de dériver vers :

```text
"AGI framework"
"autonomous super-agent"
"multi-agent swarm"
"self-improving AGI"
```

Évite également toute dérive de scope vers :

```text
JEV
un fallback cloud de décision
un second judge générique
une abstraction multi-provider créée sans besoin expérimental
```

Laya est la seule brique de décision spécialisée étudiée dans ce POC.

Ce seraient des slogans, pas des résultats.

Le projet doit rester :

```text
mesurable
falsifiable
reproductible
modulaire
simple à ablater
```

---

# 49. Résultat attendu de ta première passe

Après inspection du dépôt et avant d'écrire beaucoup de code, crée :

```text
docs/IMPLEMENTATION_PLAN.md
```

Il doit contenir :

- architecture finale proposée ;
- composants ;
- APIs/interfaces ;
- schémas de données ;
- choix du sandbox ;
- stratégie Laya ;
- stratégie de labels ;
- stratégie de benchmark ;
- risques techniques ;
- milestones ;
- ordre exact d'implémentation.

Puis commence immédiatement la **Milestone 1** sauf si tu découvres un blocage technique réel.

Ne me demande pas de choisir entre plusieurs détails mineurs que tu peux résoudre rationnellement.

Quand plusieurs options sont possibles :

1. choisis la plus simple ;
2. explique le compromis ;
3. garde l'interface extensible.

---

# 50. Définition du succès technique de Milestone 1

Je dois pouvoir exécuter quelque chose comme :

```bash
git clone ...
cd laya-dynamics-agent
uv sync
cp .env.example .env
# ajouter la clé OpenAI
uv run lda doctor
uv run lda demo
uv run lda benchmark --suite smoke
```

et obtenir un rapport montrant au minimum :

```text
GPT-only
vs
GPT + base Laya
```

avec les trajectoires complètes enregistrées.

---

# 51. Principe final

Ne construis pas :

```text
GPT + Laya parce que "deux modèles valent mieux qu'un"
```

Construis :

```text
GPT Sol = orchestrateur / générateur / raisonneur principal
Luna = sous-agents rapides pour tâches parallélisables et vérifiables
Laya = prédicteur rapide spécialisé de propriétés de transition
environment = source de vérité
trajectory log = mémoire expérimentale
prediction error = signal d'apprentissage
benchmark = arbitre final
```

Il n'existe **aucun fallback JEV** dans cette architecture.

Si Laya n'apporte pas de bénéfice mesurable, la baseline de repli est simplement :

```text
GPT-only
```

ou :

```text
GPT candidate generation + heuristic policy
```

C'est précisément ce que les ablations doivent permettre de déterminer.

La question n'est jamais :

> « Est-ce que l'architecture semble intelligente ? »

La question est :

> **« Est-ce qu'elle prédit mieux, agit mieux, coûte moins, ou apprend mieux que les baselines ? »**

Commence par auditer le repo Laya actuel, écrire `docs/IMPLEMENTATION_PLAN.md`, puis implémenter la Milestone 1.

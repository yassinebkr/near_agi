# Vers un agent avec modèle de dynamique abstraite — LLM + Laya post-entraîné

## 1. Point de départ

La question initiale était la suivante :

> Pourquoi les entreprises continuent-elles à pousser les LLM vers l'AGI alors qu'un LLM autoregressif semble, à lui seul, insuffisant pour produire une intelligence générale ?

La conclusion importante est qu'il n'existe aujourd'hui ni preuve que le scaling des LLM conduira nécessairement à l'AGI, ni preuve qu'il en est incapable.

Le paradigme actuel évolue déjà au-delà du simple :

```text
LLM plus gros -> meilleures performances
```

vers des systèmes composés de plusieurs briques :

```text
foundation model
+ reinforcement learning
+ outils
+ recherche
+ mémoire
+ planification
+ agents
+ modèles prédictifs de l'environnement
+ interaction avec le monde
```

L'hypothèse intéressante n'est donc pas nécessairement :

```text
un énorme LLM = AGI
```

mais plutôt :

```text
un système cognitif = plusieurs sous-systèmes spécialisés
```

---

# 2. Pourquoi le concept de « world model » est souvent mal compris

Le terme *world model* donne facilement l'impression d'un simulateur visuel interne :

```text
agent virtuel
+ ordinateur virtuel
+ environnement 3D
+ simulation détaillée du futur
```

Ce serait inutilement coûteux pour énormément de tâches abstraites.

Un world model utile à un agent général peut être beaucoup plus simple.

La formulation minimale est :

\[
(s_t, a_t) \rightarrow \hat{s}_{t+1}
\]

avec :

- \(s_t\) : état actuel ;
- \(a_t\) : action envisagée ;
- \(\hat{s}_{t+1}\) : état futur prédit.

L'état n'a pas besoin d'être une image ou une scène 3D.

Il peut être une représentation abstraite de :

- ce que l'agent sait ;
- ce qu'il ignore ;
- son objectif ;
- les hypothèses qu'il considère ;
- les outils disponibles ;
- les actions déjà réalisées ;
- le niveau de confiance de certaines informations ;
- l'état d'un environnement logiciel ;
- les conséquences probables d'une action.

---

# 3. Exemple : recherche sur Internet

Supposons que l'agent doive trouver la tension maximale d'un composant.

L'état peut être :

```yaml
goal:
  find_absolute_max_voltage: true

knowledge:
  manufacturer: "Texas Instruments"
  max_voltage: null

evidence:
  forum_result_found: true
  primary_datasheet_found: false

confidence:
  voltage_value: 0.25
```

Actions possibles :

```text
A1 = ouvrir un résultat de forum
A2 = rechercher le site du fabricant
A3 = ouvrir la datasheet officielle
A4 = répondre immédiatement
```

Un world model abstrait pourrait estimer :

```text
A1 -> information possible mais source secondaire
A2 -> probabilité élevée d'atteindre une source primaire
A3 -> forte probabilité de réduire l'incertitude
A4 -> risque élevé de répondre avec une information insuffisamment vérifiée
```

Il ne simule donc pas :

```text
main -> souris -> clavier -> navigateur -> pixels
```

Il simule :

```text
état de connaissance + action
-> état de connaissance probable
```

---

# 4. « Simuler toutes les possibilités » n'est pas nécessaire

Une intelligence générale n'aurait probablement pas intérêt à simuler exhaustivement toutes les possibilités.

L'approche plus réaliste est :

```text
générer quelques actions plausibles
        |
        v
prédire leurs conséquences
        |
        v
évaluer les branches intéressantes
        |
        v
approfondir seulement certaines branches
```

Soit :

\[
s_t
\rightarrow
\{a_1,a_2,a_3,\ldots\}
\]

puis :

\[
(s_t,a_i)\rightarrow \hat{s}_{t+1}^{(i)}
\]

On obtient une forme de planification contrefactuelle :

> « Si je fais X, qu'est-ce qui devrait probablement se produire ? »

---

# 5. Plusieurs niveaux de « world model »

Pour un agent général, il est probablement plus utile de parler de **modèles de dynamique** que d'un seul world model universel.

## 5.1 Monde physique

```text
je pousse l'objet
-> il tombe
-> il peut se casser
```

Très utile en robotique.

## 5.2 Monde logiciel

```text
git reset --hard
-> modifications locales perdues
```

```text
POST /purchase
-> transaction potentiellement créée
```

## 5.3 Monde informationnel

```text
ouvrir une documentation officielle
-> nouvelle information
-> diminution probable de l'incertitude
```

## 5.4 Monde causal

```text
modifier ce paramètre
-> augmente la température
-> peut réduire la stabilité
```

## 5.5 Monde social

```text
envoyer ce message
-> réponse probable
-> nouvelle information attendue
```

## 5.6 État épistémique

L'agent doit également pouvoir modéliser :

```text
ce que je sais
ce que je crois
ce que je suppose
ce que j'ignore
ce que je dois vérifier
```

C'est particulièrement important pour une AGI hypothétique.

---

# 6. Le point réellement intéressant : prediction -> action -> observation

Une architecture plus riche qu'un simple LLM pourrait fonctionner ainsi :

```text
observation
    |
    v
state encoder
    |
    v
s_t
    |
    +----> génération d'actions candidates
    |
    +----> prédiction de leurs conséquences
    |
    +----> estimation de valeur / risque / information gain
    |
    v
sélection d'une action
    |
    v
environnement réel
    |
    v
s_{t+1}
```

Puis comparer :

\[
\hat{s}_{t+1}
\]

à :

\[
s_{t+1}
\]

Le signal fondamental devient :

\[
e_t = s_{t+1} - \hat{s}_{t+1}
\]

ou, plus généralement, une erreur de prédiction structurée.

C'est cette boucle qui permet au système de corriger progressivement son modèle.

---

# 7. Pourquoi Laya est intéressant dans ce contexte

Laya est un modèle non autoregressif orienté décision.

Au moment de cette conception, le checkpoint anglais principal utilise un backbone ModernBERT-large et un decision head, pour environ 421 M de paramètres.

Il traite notamment des décisions typées :

- `choice`
- `score`
- `noul` / oui-non probabiliste

Il est conçu pour évaluer plusieurs décisions dans un seul passage plutôt que générer une réponse token par token.

L'intérêt n'est donc pas de remplacer le LLM génératif.

La séparation proposée est :

```text
LLM
-> invente / propose / raisonne / formule

Laya
-> classe / score / filtre / estime rapidement
```

Cette différence est fondamentale.

---

# 8. Laya n'est pas naturellement un world model

Laya réalise plutôt une fonction du type :

\[
f(s,q)\rightarrow P(y)
\]

alors qu'un vrai modèle de transition chercherait :

\[
f(s,a)\rightarrow \hat{s}_{t+1}
\]

Il ne faut donc pas prétendre que Laya est déjà un world model.

En revanche, on peut le post-entraîner pour apprendre des **propriétés des transitions**.

Par exemple :

```text
Will this action reveal useful information?
Will this action reduce uncertainty?
Will this action advance the goal?
Is this action reversible?
Can this action modify external state?
Is this action risky?
Will another observation probably be required?
Is the current hypothesis likely to survive this action?
```

On peut alors approximer :

\[
P(\Delta s \mid s,a)
\]

à travers un ensemble de décisions typées.

---

# 9. Architecture proposée

```text
                         +------------------+
                         |   Observation    |
                         +---------+--------+
                                   |
                                   v
                         +------------------+
                         |  State Builder   |
                         +---------+--------+
                                   |
                                  s_t
                                   |
               +-------------------+-------------------+
               |                                       |
               v                                       v
        +--------------+                       +----------------+
        | GPT / LLM    |                       | episodic memory|
        | proposer     |                       | trajectories   |
        | des actions  |                       +----------------+
        +------+-------+
               |
        {a1,a2,a3,...}
               |
               v
        +------------------------------------------+
        |        Laya post-entraîné                |
        |                                          |
        | P(goal_progress)                         |
        | P(success)                               |
        | P(information_gain)                      |
        | P(risk)                                  |
        | P(reversible)                            |
        | P(needs_observation)                     |
        +------------------+-----------------------+
                           |
                           v
                  Planner / policy
                           |
                           v
                      action a*
                           |
                           v
                    environnement
                           |
                           v
                         s_t+1
                           |
              +------------+-------------+
              |                          |
              v                          v
      prediction error             trajectory log
              |                          |
              +------------+-------------+
                           |
                           v
                     training data
```

---

# 10. Extension plus ambitieuse

Dans une seconde phase, on pourrait ajouter une représentation latente de transition :

\[
f(s_t,a_t)\rightarrow\Delta z
\]

puis :

\[
\hat z_{t+1} = z_t + \Delta z
\]

Cela permettrait théoriquement de dérouler plusieurs étapes sans agir réellement :

\[
z_t
\xrightarrow{a_1}
\hat z_{t+1}
\xrightarrow{a_2}
\hat z_{t+2}
\]

Cette partie est nettement plus expérimentale.

Le premier POC ne doit pas en dépendre.

---

# 11. Pourquoi cette architecture peut être meilleure qu'un « judge » générique

Une expérience précédente avec JEV dans un agent Shopify a donné des résultats mitigés.

Le problème général d'un *judge* ajouté au milieu d'un agent est qu'il peut :

- répéter approximativement ce que le LLM sait déjà faire ;
- introduire de la latence ;
- ajouter un second point de défaillance ;
- bloquer de bonnes décisions ;
- être peu corrélé avec la réussite réelle de la tâche ;
- ne jamais apprendre des trajectoires de l'agent ;
- augmenter le coût API s'il est distant.

Notre utilisation de Laya doit donc être différente.

## Mauvais objectif

```text
GPT propose une action
-> Laya dit arbitrairement oui/non
```

## Objectif proposé

```text
GPT propose plusieurs actions
-> Laya prédit des propriétés mesurables
   de leurs conséquences
-> le planner choisit
-> on exécute
-> on observe le résultat réel
-> on mesure l'erreur
-> on enrichit le dataset
-> on réentraîne
```

Laya devient alors une **fonction apprise de dynamique/valeur locale** plutôt qu'un simple juge.

---

# 12. Pourquoi le local est particulièrement intéressant

Dans cette architecture, Laya peut être interrogé très souvent :

```text
10 actions candidates
x
6 propriétés
x
plusieurs étapes
```

Un service API payant devient rapidement coûteux et ajoute de la latence réseau.

Une inférence locale apporte :

- absence de coût API par décision ;
- faible latence ;
- confidentialité ;
- possibilité de batcher beaucoup de décisions ;
- contrôle total du checkpoint ;
- possibilité de post-entraînement ;
- reproductibilité ;
- fonctionnement hors ligne.

D'après le retour pratique sur la machine cible, le modèle Laya complet tient déjà en VRAM avec une empreinte mémoire faible en inférence. Le POC peut donc partir directement du modèle complet au lieu d'être conçu autour d'un checkpoint réduit.

---

# 13. Le dataset réellement important

Le dataset ne doit pas être simplement composé de préférences humaines.

L'unité fondamentale doit être :

```text
(state_before, action, state_after, outcome)
```

Exemple :

```json
{
  "state_before": {
    "goal": "find official maximum voltage",
    "primary_source_found": false,
    "confidence": 0.31
  },
  "action": {
    "tool": "open_url",
    "target": "manufacturer_datasheet"
  },
  "state_after": {
    "primary_source_found": true,
    "confidence": 0.94
  },
  "outcome": {
    "goal_progress": 0.92,
    "information_gain": 0.88,
    "external_side_effect": false,
    "success": true
  }
}
```

Les labels peuvent en partie être générés automatiquement à partir des deltas entre états.

---

# 14. Important : empêcher les fuites de labels

Le modèle ne doit pas apprendre des raccourcis triviaux.

Exemples de mauvaises corrélations :

```text
si history != empty -> DONE
```

ou :

```text
si texte contient "success" -> action réussie
```

Il faudra :

- séparer les sites/tâches entre train et test ;
- introduire des contre-exemples ;
- éviter les templates trop répétitifs ;
- tester sur des trajectoires inconnues ;
- faire des ablations ;
- mesurer la calibration et pas seulement l'accuracy.

La documentation officielle de Laya sur le fine-tuning d'un browser-agent insiste justement sur l'importance du format d'entrée, des vrais états `DONE`, des négatifs intermédiaires et des corrections on-policy.

---

# 15. POC recommandé

Le premier prototype doit rester contrôlable.

## Environnement

Un mini-environnement web local ou sandboxé avec des tâches telles que :

```text
trouver une information
naviguer vers une page
remplir un formulaire
comparer deux informations
identifier une source primaire
corriger une hypothèse après observation
```

## Baselines

### A. GPT seul

```text
state -> GPT -> action
```

### B. GPT + Laya zero-shot

```text
state -> GPT candidates -> base Laya -> action
```

### C. GPT + Laya post-entraîné

```text
state -> GPT candidates -> specialised Laya -> action
```

### D. Optionnel : GPT + JEV

Seulement si l'intégration précédente est suffisamment facile à reproduire proprement.

Cette baseline serait intéressante pour comparer :

- cloud vs local ;
- générique vs spécialisé ;
- coût ;
- latence ;
- qualité de décision.

---

# 16. Métriques

Le POC doit produire des mesures quantitatives.

## Réussite

\[
TaskSuccessRate
\]

## Efficacité

```text
actions nécessaires
actions inutiles
retries
loops
tool errors
```

## Latence

```text
LLM latency
Laya latency
planner latency
total task latency
```

## Coût

```text
input tokens
output tokens
API cost
local GPU time
```

## Qualité prédictive

```text
Brier score
log loss
accuracy
top-k accuracy
ECE / calibration error
```

## Modèle de transition

Comparer les prédictions :

```text
predicted:
goal_progress = 0.84

observed:
goal_progress = 0.72
```

et suivre l'erreur au fil des réentraînements.

---

# 17. Critère de réussite du projet

L'expérience doit pouvoir échouer.

C'est essentiel.

Le résultat intéressant n'est pas :

> « nous avons construit un embryon d'AGI »

mais :

> « nous avons testé si un petit decision model local spécialisé sur les transitions réelles d'un agent permet à un LLM planner de prendre de meilleures décisions avec moins d'actions et moins de coût. »

Une première réussite convaincante serait par exemple :

```text
TaskSuccess >= baseline GPT seul
ET
actions inutiles diminuent
ET
coût API diminue ou reste stable
ET
latence ajoutée par Laya reste faible
ET
les scores de Laya sont réellement calibrés
ET
le fine-tuning dépasse nettement le zero-shot
```

---

# 18. Hypothèse scientifique

## H0 — hypothèse nulle

> Ajouter Laya spécialisé n'améliore pas suffisamment le comportement de l'agent pour justifier sa complexité.

## H1 — hypothèse expérimentale

> Un petit modèle non génératif local, entraîné sur les transitions réelles de l'agent, peut apprendre des propriétés utiles de la dynamique de tâche et améliorer la planification d'un LLM génératif.

---

# 19. Étapes proposées

## Phase 0 — audit

- reproduire Laya local ;
- mesurer VRAM / RAM / latence ;
- reproduire ses sorties typées ;
- inspecter son format de fine-tuning ;
- documenter le hardware et les versions.

## Phase 1 — environnement déterministe

- environnement web local ;
- état explicite ;
- actions explicites ;
- logs reproductibles ;
- GPT comme planner.

## Phase 2 — collecte

Enregistrer :

```text
s_t
candidate_actions
selected_action
predictions
actual_s_t+1
task_outcome
```

## Phase 3 — Laya zero-shot

Mesurer objectivement s'il apporte quelque chose.

## Phase 4 — post-entraînement

Construire un dataset à partir des vraies trajectoires.

## Phase 5 — calibration

Calibrer les probabilités sur un jeu dédié.

## Phase 6 — benchmark aveugle

Tester sur :

- nouvelles tâches ;
- nouveaux sites/environnements ;
- nouvelles formulations.

## Phase 7 — DAgger / on-policy

Récupérer les erreurs commises par le modèle lui-même.

Les ajouter au dataset.

Réentraîner.

## Phase 8 — planning multi-step

Seulement après validation du prédicteur one-step.

---

# 20. Ce qu'il ne faut pas faire au début

Ne pas commencer par :

- un navigateur Internet totalement libre ;
- un système multi-agent complexe ;
- une mémoire vectorielle élaborée ;
- un world model vidéo ;
- du RL end-to-end ;
- un latent transition model non interprétable ;
- Shopify en production ;
- des actions irréversibles ;
- des dizaines de tools.

Cela rendrait impossible de savoir quelle brique améliore ou dégrade le système.

---

# 21. Connexion future avec l'agent Shopify

Une fois l'hypothèse validée dans le sandbox, le système pourrait être appliqué à l'agent Shopify existant.

Exemple d'état :

```yaml
goal:
  improve_homepage_conversion: true

store:
  theme: "..."
  unsaved_changes: true

observations:
  mobile_layout_issue: true
  broken_asset: false

history:
  - inspect_theme
  - read_homepage
```

Actions candidates :

```text
edit_section
inspect_css
preview_mobile
publish_theme
ask_user
```

Laya spécialisé pourrait estimer :

```text
goal_progress
risk
reversibility
need_for_observation
probability_of_success
```

Point crucial :

```text
publish_theme
```

et :

```text
inspect_css
```

ne doivent évidemment pas avoir le même profil de risque.

---

# 22. Principe architectural final

Le but n'est pas :

```text
LLM + un deuxième modèle qui vote
```

Le but est :

```text
GENERATIVE MODEL
    |
    | propose
    v
candidate actions
    |
    v
LEARNED DYNAMICS / VALUE MODEL
    |
    | predicts consequences
    v
planner
    |
    v
real action
    |
    v
observation
    |
    v
prediction error
    |
    v
learning
```

C'est cette boucle qui transforme l'expérience en quelque chose de beaucoup plus intéressant qu'un simple système de classification.

---

# 23. Références techniques utiles

- NandhaKishorM/laya — dépôt principal Laya.
- Documentation Laya — typed `choice`, `score`, `noul`, fine-tuning et calibration.
- `docs/finetune_browser_agent.md` — exemple particulièrement pertinent de spécialisation de Laya pour la sélection d'actions d'un browser agent.
- Mind2Web — dataset de trajectoires web, potentiellement utile comme source complémentaire.
- DAgger — idée utile pour ajouter des corrections sur les états réellement rencontrés par la politique actuelle.

---

# Conclusion

Le projet ne consiste pas à prétendre construire une AGI.

Il consiste à isoler une question beaucoup plus sérieuse :

> **Un modèle local rapide et non génératif comme Laya, post-entraîné sur les conséquences réelles des actions d'un agent, peut-il devenir un prédicteur de dynamique abstraite suffisamment utile pour améliorer la planification d'un LLM ?**

Si la réponse est oui, cela fournit une brique intéressante :

```text
LLM génératif
+
decision/dynamics model local
+
mémoire de trajectoires
+
boucle prediction -> action -> observation -> correction
```

Ce serait déjà un résultat substantiel.

# imagegen

Génération d'images en local, à partir d'un prompt et **optionnellement** d'une
image source, avec des modèles libres et téléchargeables. Aucun service payant,
aucune clé d'API obligatoire.

Cette première étape ne contient que la **partie IA** : une bibliothèque Python
et une interface en ligne de commande. Pas d'interface graphique, pas de serveur.

```
imagegen generate "un renard roux dans une foret enneigee"
imagegen generate "le meme, en hiver polaire" -i photo.png --strength 0.6
```

---

## Installation

Prérequis : Python 3.10+ et [uv](https://docs.astral.sh/uv/) (ou pip).

PyTorch **n'est pas** dans les dépendances : il doit venir de l'index
correspondant à votre matériel, sinon pip installe une version CPU qui
désactive silencieusement l'accélération.

```powershell
uv venv --python 3.12 .venv

# GPU NVIDIA (adapter cu128 / cu130 selon le pilote)
uv pip install --python .venv\Scripts\python.exe `
  --index-url https://download.pytorch.org/whl/cu128 torch torchvision

uv pip install --python .venv\Scripts\python.exe -e .
```

Vérifier que tout est en place :

```powershell
.venv\Scripts\imagegen doctor
```

`doctor` contrôle notamment que la version de PyTorch installée contient bien
les noyaux de votre GPU — `torch.cuda.is_available()` renvoie `True` même
lorsque ce n'est pas le cas, et l'erreur n'apparaît alors qu'au bout de
quarante secondes de génération.

---

## Utilisation

### Générer depuis un prompt

```powershell
imagegen generate "un phare dans la tempete, peinture a l'huile"
imagegen generate "un portrait au crayon" -m sd15 -W 512 -H 768 --seed 42
```

### Générer depuis un prompt **et** une image

C'est la même commande : ajouter `-i` bascule en image-to-image.

```powershell
imagegen generate "style cyberpunk neon" -i photo.png --strength 0.7
```

`--strength` dose la transformation : `0.3` reste proche de l'image source,
`0.9` s'en éloigne fortement.

Attention à une subtilité : l'image-to-image n'exécute que
`int(etapes x strength)` étapes de débruitage. À `--steps 1 --strength 0.5`,
c'est **zéro** étape, et diffusers renvoie alors l'image source inchangée sans
la moindre erreur. `imagegen` refuse ce cas et indique la valeur à utiliser.

### Régler l'effort

Un seul réglage arbitre qualité contre temps :

```powershell
imagegen generate "un renard roux" --effort draft   # le plus rapide
imagegen generate "un renard roux"                  # balanced, par défaut
imagegen generate "un renard roux" --effort max     # le meilleur rendu
```

Ce n'est pas un simple multiplicateur d'étapes, parce que « plus d'étapes »
n'est pas toujours mieux : sur un modèle distillé, augmenter les étapes sans
changer l'adaptateur **dégrade** l'image, et avec un échantillonneur ancestral
cela re-tire une image différente au lieu de l'affiner. Chaque palier déplace
donc un ensemble cohérent — adaptateur, étapes, échantillonneur, guidage, passe
d'affinage.

Sur le modèle par défaut, toute l'échelle tourne sur **les mêmes poids SDXL** :

| niveau | étapes | guidage | adaptateur | affinage |
|---|---|---|---|---|
| `draft` | 2 | 0 | Lightning 2 étapes | — |
| `fast` | 4 | 0 | Lightning 4 étapes | — |
| `balanced` | 8 | 0 | Lightning 8 étapes | — |
| `high` | 28 | 7 | aucun (SDXL de base) | — |
| `max` | 28 | 7 | aucun (SDXL de base) | 14 étapes |

Les paliers bas échangent une LoRA de 394 Mo ; les deux derniers la retirent
pour retrouver le vrai guidage sans classifieur. Aucun palier ne demande un
second modèle. `imagegen models show <clé>` affiche l'échelle de n'importe quel
modèle.

Les options explicites restent prioritaires : `--effort max --steps 6` fait
bien six étapes. Le niveau ne remplit que ce qui n'a pas été fixé. Pour fixer le
niveau une bonne fois : `$env:IMAGEGEN_EFFORT = "high"`.

La passe d'affinage de `max` peut travailler sur une image agrandie — un
« hires fix ». Elle est livrée **sans agrandissement**, sur mesure : sur la
carte de référence (8 Go), agrandir à 1,25× a fait passer la génération de 38 s
à plus de sept minutes, parce que la toile plus grande déborde de la VRAM et que
Windows ne lève pas d'erreur, il déborde en RAM système. Le mécanisme est
conservé et `Backend.can_afford_upscale` décide à l'exécution : sur une carte
de 12 Go, remonter `MAX_REFINE_SCALE` est une ligne à changer.

### Prompts en français

Les encodeurs de texte de SD et SDXL sont entraînés en anglais. Un prompt
français n'échoue pas — il perd silencieusement les mots que CLIP ne sait pas
représenter. Mesuré sur ce projet, avec le modèle par défaut :

> « un verre d'eau et **une clé en laiton** sur une table en bois sombre »
> → deux verres, **aucune clé**.
> La même demande en anglais rend correctement le verre et la clé en laiton.

`imagegen` détecte donc un prompt non anglais et le traduit avant de l'envoyer
au modèle, en affichant toujours la traduction — la traduction automatique a ses
propres défauts (« visage buriné » revient en *burin face*) et vous devez voir
d'après quoi l'image a été produite.

```powershell
imagegen generate "un phare dans la tempête"     # traduit, traduction affichée
imagegen generate "..." --translate never        # envoie le français tel quel
```

Détail qui compte : **gardez vos accents**. « tempête » se traduit par *storm*,
« tempete » par *temple*.

Le fichier PNG conserve le prompt d'origine à côté de celui réellement utilisé.

Ce que la mesure dit, sans enjoliver : sur 6 prompts × 3 graines, appariés,
notés par CLIP contre la référence anglaise, la traduction gagne en moyenne
+0,5 point — un écart que l'intervalle de confiance ne distingue pas du bruit à
cet effectif. Le détail par prompt est plus net : **+3,8 sur la clé en laiton**
(l'image traduite est identique au pixel près à l'anglaise), **+2,6 sur le
portrait**, et −3,0 sur la méduse, où la version traduite en montre plusieurs au
lieu d'une — une image correcte que le score pénalise. La traduction répare
donc les vrais échecs (objets absents) et reste neutre ailleurs ; c'est pour ça
qu'elle est active par défaut, et affichée pour que vous puissiez la contredire.

### Explorer

| Commande | Rôle |
|---|---|
| `imagegen models list` | Les modèles disponibles, leur licence, leur poids |
| `imagegen models show sdxl-lightning` | Le détail d'un modèle |
| `imagegen devices` | Les accélérateurs détectés (GPU, NPU, CPU) |
| `imagegen backends` | Les moteurs d'inférence et pourquoi ils sont, ou non, utilisables |
| `imagegen doctor` | Diagnostic complet (`--smoke` ajoute une vraie génération de test) |
| `imagegen download sdxl-lightning` | Pré-télécharger les poids |

Toutes les commandes acceptent `--json` pour une sortie exploitable par script.

---

## Modèles

Le modèle par défaut est **`sdxl-lightning`** (SDXL + adaptateur Lightning en
4 étapes). C'est le seul modèle 1024 px qui tienne dans 8 Go **sans
quantification**, sous licence permettant l'usage commercial, et sans jeton
Hugging Face.

| clé | taille | étapes | téléch. | licence | usage commercial |
|---|---|---|---|---|---|
| `sdxl-lightning` | 1024² | 4 | 7,3 Go | OpenRAIL++-M | oui |
| `sdxl` | 1024² | 30 | 7,0 Go | OpenRAIL++-M | oui |
| `sd15` | 512² | 25 | 2,7 Go | OpenRAIL-M | oui |
| `sd15-lcm` | 512² | 4 | 2,9 Go | OpenRAIL-M | oui |
| `lcm-dreamshaper` | 768² | 4 | 2,6 Go | MIT | oui |
| `flux2-klein` | 1024² | 4 | 15,8 Go | Apache-2.0 | oui |

`imagegen models list --all` ajoute les modèles réservés à un usage explicite :
ceux dont la licence est **non commerciale** (`sdxl-turbo`, `sd-turbo`,
`flux1-kontext`), ceux qui exigent un jeton Hugging Face malgré une licence
libre (`flux1-schnell` est Apache-2.0 **et** sous accord), et ceux qui
demandent une quantification pour tenir en 8 Go.

Le catalogue est purement déclaratif : `imagegen models` répond
instantanément, sans importer PyTorch ni contacter le réseau.

---

## Ce que l'architecture prévoit pour le NPU

L'exécution doit pouvoir basculer du GPU vers un NPU. Le code est découpé pour
que cela devienne l'ajout d'un *backend*, sans toucher au générateur ni à la
CLI. Concrètement, quatre contraintes propres aux NPU sont déjà absorbées :

**Les formes et le guidage font partie de la clé de compilation.** Tous les
chemins NPU actuels compilent le débruiteur pour une résolution et une taille de
lot figées — et le guidage sans classifieur (CFG) double ce lot, ce qui explique
que `guidance_scale` soit un argument de `reshape()` chez OpenVINO. Ces valeurs
vivent donc dans `PipelineSpec` (la clé de compilation), séparée de
`GenerationRequest` (les paramètres d'appel). Un backend déclare dans
`recompiles_on` ce qui l'oblige à recompiler ; le générateur réutilise le
pipeline chargé quand `accepts()` le permet, et signale la recompilation comme
un temps distinct de la génération.

**Le placement est par composant.** Le seul chemin NPU Intel qui fonctionne
aujourd'hui est hétérogène : encodeur de texte sur CPU, UNet sur NPU, VAE sur
GPU. Un backend reçoit donc un `DevicePlan`, pas un périphérique.

**L'image-to-image dépend de l'artefact, pas du backend.** Il faut un *encodeur*
VAE : `optimum-cli export onnx` ne le produit que pour la tâche img2img, et les
paquets précompilés de Qualcomm ne livrent que le décodeur. `supports_img2img`
est donc **déduit** des composants réellement présents, jamais déclaré.

**Les schedulers ne sont pas portables.** L'ensemble exposé par OpenVINO est
fermé et ne contient ni DPM++ ni UniPC. L'API publique utilise donc une
énumération neutre (`SchedulerKind`), que chaque backend traduit vers ce qu'il
possède, en signalant les dégradations.

État actuel des backends :

| backend | état |
|---|---|
| `torch` | **complet** — CUDA, XPU, MPS, CPU |
| `openvino` | écrit, **non validé** sur NPU (nécessite `imagegen[openvino]`) |
| `onnx` | écrit, **non validé** sur NPU (nécessite `imagegen[onnx-*]`) |

La détection matérielle, elle, fonctionne dès maintenant pour tous :
`imagegen devices` liste les NPU Intel, Qualcomm et AMD s'ils sont présents, et
`imagegen backends` distingue trois états que l'on confond souvent — paquet
absent, paquet installé mais compilé sans le provider, et matériel absent.

Une honnêteté nécessaire : la génération d'images sur NPU n'est pas une
fonctionnalité officiellement supportée chez Intel (leur documentation « GenAI
sur NPU » ne liste que les LLM, VLM et Whisper), et les chemins Qualcomm et AMD
n'acceptent que des modèles précompilés par le constructeur. Ces backends sont
étiquetés « expérimental » tant que personne ne les a fait tourner sur le
matériel correspondant.

---

## Notes de performance (RTX 5050, 8 Go)

Mesuré sur la machine de référence, GPU exclusif, modèle par défaut en 1024².
Deux colonnes, parce que l'écart entre les deux est la mesure la plus utile :
la première image d'une session paie le chargement et la compilation des noyaux
CUDA, les suivantes non.

| effort | étapes | 1re image | images suivantes | pic VRAM |
|---|---|---|---|---|
| `draft` | 2 | ~27 s | ~5,2 s | 5,8 Go |
| `fast` | 4 | ~38 s | ~5,7 s | 5,8 Go |
| `balanced` | 8 | ~38 s | ~6,9 s | 5,8 Go |
| `high` | 28 + CFG | ~43 s | ~21 s | 5,6 Go |

Le résultat le plus contre-intuitif : **le coût par étape est négligeable
devant le coût fixe par appel**. Passer de 4 à 8 étapes coûte 1,2 s ; le
transfert d'offload d'environ 5 Go sur le PCIe en coûte 4,5 à chaque appel.
C'est pourquoi le palier par défaut est à 8 étapes et non à 4 : la qualité est
visiblement meilleure pour un surcoût marginal.

Le guidage sans classifieur (CFG) double le batch du débruiteur et coûte 1,5× —
pas le facteur 60 qu'une première mesure semblait montrer. Cette mesure-là était
faussée par un autre processus qui se partageait le GPU ; le pic VRAM reste
autour de 5,6 Go dans tous les cas, sans aucun débordement.

Pour `sd15` en 512², 25 étapes : ~9 s la première image.

Sur 8 Go, `imagegen` active automatiquement l'offload par sous-modèle pour
SDXL : les poids fp16 font 7 Go, le bureau Windows en consomme déjà plus d'un.
Le choix est expliqué dans la sortie et forçable avec `--no-cpu-offload` /
`--cpu-offload` / `--sequential-offload`.

Cela vaut la peine d'insister : sur Windows, **dépasser la VRAM ne provoque pas
d'erreur**. Le pilote WDDM déborde silencieusement en RAM système. Mesuré ici,
SDXL forcé entièrement en VRAM prend **190 s au lieu de 13 s**. C'est pourquoi
la décision est prise avant le chargement, à partir de la VRAM réellement
libre, plutôt qu'en attendant un `OutOfMemoryError` qui ne viendra pas.

---

## Ce qui a été mesuré, et ce que ça a changé

Les défauts de ce projet viennent de mesures sur la machine de référence, pas
d'intuitions. Les cinq qui ont changé le code :

**Le coût par étape est négligeable devant le coût fixe par appel.** 4 étapes
coûtent 5,7 s, 8 étapes en coûtent 6,9 : le transfert d'offload de ~5 Go domine.
Le palier par défaut est donc passé de 4 à 8 étapes — la fourrure et la neige
sont visiblement plus détaillées pour 1,2 s de plus.

**Un prompt français perd des mots.** « une clé en laiton » disparaît
complètement de l'image. D'où la traduction automatique par défaut.

**`unload()` ne libérait pas la VRAM.** `enable_model_cpu_offload` installe des
hooks accelerate qui gardent une référence sur chaque sous-modèle : sans
`remove_all_hooks`, un second chargement dans le même processus laissait 7,9 Go
occupés sur 8, et la génération suivante prenait 38 s au lieu de 7. C'est le
chemin exact qu'emprunte un changement de modèle ou de niveau d'effort.

**L'estimation mémoire était 3× trop pessimiste.** Le pic réel mesuré sous
offload est de 5,4 à 5,6 Go quelle que soit la configuration ; l'estimateur en
annonçait 1,5 Go de plus que la réalité, ce qui pouvait faire basculer sur
l'offload séquentiel — « très lent » selon la documentation de diffusers — sans
raison. Être pessimiste ici n'est pas prudent, c'est nuisible.

**Une première mesure était fausse.** `high` semblait prendre 524 s. En réalité
un autre processus se partageait le GPU ; isolé, c'est 38 s. Les mesures GPU de
ce projet sont refaites avec la carte en exclusivité, et le protocole est
apparié — mêmes prompts, mêmes graines dans chaque bras — parce que la variance
entre prompts est du même ordre que l'effet recherché.

## Reproductibilité

Chaque image est nommée d'après sa graine, et porte ses paramètres dans les
métadonnées PNG (`--sidecar` ajoute un `.json`).

À paramètres identiques, la même graine redonne exactement la même image. Deux
limites, mesurées et non supposées :

- **En changeant `-n`**, l'image reste la même mais pas au pixel près. Le bruit
  de départ est identique, mais un lot de 3 ne sélectionne pas les mêmes noyaux
  matriciels qu'un lot de 1, et l'addition flottante n'est pas associative.
  Écart moyen mesuré : 2,5/255, contre 73/255 pour une graine différente.
- **En changeant de backend**, une graine ne correspond jamais : les chemins NPU
  imposent une compression des poids en int8/int4, et le générateur aléatoire
  diffère.

Les générateurs sont créés sur CPU, jamais sur GPU : le flux aléatoire CUDA
n'est pas stable d'une carte ou d'un pilote à l'autre.

---

## Développement

```powershell
.venv\Scripts\python -m pytest tests          # rapide, hors ligne
.venv\Scripts\python -m pytest tests -m slow  # bout en bout, télécharge et génère
.venv\Scripts\python -m ruff check src tests
```

Organisation :

```
src/imagegen/
  types.py          PipelineSpec (clé de compilation) / GenerationRequest (appel)
  errors.py         erreurs porteuses d'une action à effectuer
  hardware/         détection des accélérateurs, sans rien savoir de la diffusion
  models/registry.py catalogue déclaratif (aucun import lourd)
  backends/         base.py = le contrat ; torch / openvino / onnx
  generation/       façade, dimensionnement des images, métadonnées
  cli.py            interface terminal
```

## Licences

Le code est sous licence MIT. **Les modèles ont leur propre licence**, et
certaines sont restrictives. `imagegen models show <clé>` affiche la licence,
le droit d'usage commercial et les conditions empilées (par exemple les
conditions Gemma qui s'ajoutent à l'Apache-2.0 de SANA). Les modèles non
commerciaux ne sont jamais utilisés par défaut ni proposés sans avertissement.

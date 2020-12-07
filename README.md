# Algorithme génétique : Snake

## Fonctionnel

Le but de ce projet est la création d'un algorithme génétique adapté à la maximisation d'un score au jeu de snake.
Ainsi, le programme génère des "individus" aléatoires modélisé par de simple réseau de neurones à propagation avant. Une selection des meilleurs individus est ensuite réalisée afin d'en générer de nouveaux. Au bout de plusieurs génération, les performances des individus convergeront vers le meilleur score possible étant donné les paramètres.

## Technique

#### Installation

Toutes les libraires nécéssaires sont dans le fichier `requirements.txt`, une fois installées il suffit d'éxecuter le fichier `main.py`.

#### Exécuter le programme
Vous pouvez, dans ce fichier `main.py`, modifier les paramètres afin de visualiser leurs impact sur les résultats.
Par défaut, le code affichera l'avancement des générations dans le terminal ainsi que les scores de chaque individus. Tous les 100 générations, vous pourrez appuyer sur une touche afin de visualiser les 3 meilleurs individus dans une interface pygame.

<p align="center">
<img src="img/res.gif" width="400" height="400"/>
</p>

L'issue de l'éxecution sera variable de par le fait que l'évolution de la population est en partie aléatoire. Il vous faudra donc peut être relancer le programme plusieurs fois avant d'avoir des résultats concluants.
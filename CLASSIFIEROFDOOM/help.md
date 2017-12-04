Telecharger le dataset ici : https://homes.cs.washington.edu/~hrashkin/factcheck.html (le premier lien)

executer depedencies.sh

Lancer python3 en interactif

from detection import *

Si vous voulez reentrainer le model faire : main_train()

Si vous voulez tester le model faire :
model, vectorizer = load_model()
main_test(model, vectorizer)

Si vous voulez faire une prediction  :
model, vectorizer = load_model()
make_prediction(text, model, vectorizer) avec text = au text que vous voulez prédire





# Prédiction de l'espérance de vie à l'aide de la régression linéaire simple

## 1) Exploration des données :
- Pour explorer les données on utilise :```df = d.read_csv('Life_Expectancy_Data.csv')```
- Pour tracer l’histogramme on utilise :```sns.histplot(df['Life_expectancy'], kde=True)```
![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/ab6c2007-740e-47a3-8ea3-2bca1f5eddfe)

### Pays en développement contre pays développés

![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/07242b54-1d8d-4c84-8d2f-b4b94e467da2)
Dans l’ensemble de données, il y a plus de pays en développement que de pays développés

## 2) Traitement des valeurs aberrantes :
![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/180b6d2b-f13e-4d40-b817-510718eddb8f)
![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/0d36e8ec-294a-47cf-9ac4-94b6019815ae)

**Tous les Variables sauf Adult_Mortality**

![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/e9458d26-e2a6-47a0-95fb-09e9c8dc0cdd)

**Adult Mortality**
![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/d8e70f3b-fa73-4b92-ae6c-935bb3bc84bd)
![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/b7d10513-edae-4c35-bed3-fcb5e5065afd)

## 3) Analyse des relations bivariées
![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/00535414-7550-44f6-b4e4-e76a80bdffee)
![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/9e1002a9-e4c5-458e-9398-6f697fb52012)

## 4) Choix de la variable indépendante :
![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/4b158e63-dd70-4cf8-bb6f-a333223f5f40)

- La variable la plus corrélée avec Life_expectancy est Schooling avec un coefficient
de corrélation de 0.701367. Voici le classement des variables par ordre de
corrélation absolue (positive ou négative) avec l'espérance de vie
(Life_expectancy):
1. Schooling: 0.701367 (positive correlation)
2. Adult Mortality: -0.687521 (negative correlation)
3. Income_composition_of_resources: 0.679205 (positive correlation)
4. thinness_1-19 years: -0.475847 (negative correlation)
5. thinness_5-9 years: -0.473240 (negative correlation)
6. Alcohol: 0.408675 (positive correlation)
Ainsi, la variable Schooling a la corrélation positive la plus élevée avec l'espérance de vie,
ce qui indique qu'une augmentation du niveau de scolarisation est fortement associée à une
augmentation de l'espérance de vie. La corrélation avec Adult Mortality est également
forte, mais elle est négative, indiquant qu'une augmentation de la mortalité adulte est
associée à une diminution de l'espérance de vie.

## 5) Préparation des données :
``` import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
selected_variables_X =['Life_expectancy']
selected_variables_Y =['Schooling'] # 'Schooling' la plus corrélée
X = df[selected_variables_X]
y = df[selected_variables_Y]
#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.35, random_state=42)
```

## 6) Construction du modèle :

```#Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
```

## 7) Évaluation du modèle :
```#Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
#Calculer la MSE
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
#Calculer le R²
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2}')
```
**Output :**
Mean Squared Error: 5.264736837236674
R² Score: 0.49740820899422744

## 8) Visualisation des résultats :
```import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate,
KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
#Define the features and target variable
selected_variables_X = ['Adult Mortality', 'Alcohol', 'thinness_1-19
years', 'Income_composition_of_resources']
selected_variables_Y = ['Life_expectancy']
X = df[selected_variables_X]
y = df[selected_variables_Y]
#Initialize the model
model = LinearRegression()
#Perform 9-fold cross-validation
kf = KFold(n_splits=9, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, scoring='r2', cv=kf)
print('Cross-Validation R^2 Scores:', scores)
print('Average Cross-Validation R^2:', scores.mean())
#Print cross-validation results
cv_results = cross_validate(model, X, y, cv=kf, scoring='r2',
return_train_score=True)
print("Cross-validation scores (test):", cv_results['test_score'])
print("Cross-validation scores (train):", cv_results['train_score'])
print("Mean cross-validation score (test):",
cv_results['test_score'].mean())
#Train the model on the full dataset
model.fit(X, y)
#Make predictions on new data
X_sen = pd.DataFrame({
'Adult Mortality': [100, 200, 300, 400, 500], # Add new values
'Alcohol': [5, 6, 7, 8, 9], # Add new values
'thinness_1-19 years': [2, 3, 4, 5, 6], # Add new values
'Income_composition_of_resources': [0.5, 0.6, 0.7, 0.8, 0.9] # Add
new values
})
y_sen = model.predict(X_sen)
print("Predictions:", y_sen)
#Plot cross-validated predictions vs. actual values
y_cv_pred = cross_val_predict(model, X, y, cv=kf)
plt.figure(figsize=(10, 6))
plt.scatter(y, y_cv_pred, color='green', label='CV Predicted vs
Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red',
linestyle='--', linewidth=2, label='Ideal fit')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Cross-validated Predicted Life Expectancy')
plt.title(f'Cross-validated Actual vs Predicted Life Expectancy (R^2 =
{scores.mean():.2f})')
plt.legend()
plt.show()
```
![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/d65a411d-d161-49dd-a074-206823b294e8)

## 9) Interprétation des résultats :
```#Analyse
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
#Interprétation
print(f"L'espérance de vie augmente de {model.coef_[0]} années pour
chaque année supplémentaire de scolarisation.")
```

**Interprétation Finale :**

Les résultats de l'analyse montrent des relations significatives entre plusieurs facteurs
socio-économiques et l'espérance de vie. Voici une interprétation détaillée des résultats
obtenus :
Coefficient de détermination R^2
Les scores R^2 de validation croisée pour le modèle sont les suivants :

- Scores R^2 de validation croisée (test) : [0.6875, 0.5721, 0.6669, 0.6608, 0.6672,
0.7243, 0.7445, 0.7137, 0.7218]
- Scores R^2 de validation croisée (train) : [0.6847, 0.6996, 0.6869, 0.6878, 0.6874,
0.6800, 0.6787, 0.6815, 0.6803]
Le coefficient de détermination moyen R^2 pour le test est de 0.6843, ce qui signifie que
le modèle explique environ 68.43% de la variabilité de l'espérance de vie.
Coefficients du Modèle
Les coefficients obtenus du modèle de régression linéaire multiple sont les suivants :
- Intercept : 64.3209
- Coefficients : [-0.0354, 0.2084, -0.2944, 17.8567]
Interprétation des Coefficients
Les coefficients indiquent l'impact des variables indépendantes sur l'espérance de vie :
- Première variable (e.g., scolarisation) : Une augmentation d'une unité est associée
à une diminution de l'espérance de vie de 0.0354 années.
- Deuxième variable : Une augmentation d'une unité est associée à une
augmentation de l'espérance de vie de 0.2084 années.
- Troisième variable : Une augmentation d'une unité est associée à une diminution de
l'espérance de vie de 0.2944 années.
- Quatrième variable : Une augmentation d'une unité est associée à une
augmentation de l'espérance de vie de 17.8567 années.
Prédictions
Les prédictions du modèle pour les nouvelles observations sont les suivantes :
- Espérance de vie prédite pour la première observation : 70.1619 années
- Espérance de vie prédite pour la deuxième observation : 68.3208 années
- Espérance de vie prédite pour la troisième observation : 66.4797 années
- Espérance de vie prédite pour la quatrième observation : 64.6387 années
- Espérance de vie prédite pour la cinquième observation : 62.7976 années

<hr>

### Conclusion : 

Le modèle de régression linéaire multiple a montré que plusieurs facteurs
socio-économiques ont des impacts significatifs sur l'espérance de vie. Les résultats des
scores de validation croisée R^2 indiquent que le modèle explique bien la variabilité de
l'espérance de vie, mais il reste encore environ 32% de la variabilité inexpliquée par les
variables incluses dans le modèle.
En particulier, la quatrième variable montre un impact particulièrement fort et positif,
soulignant l'importance des ressources économiques dans l'amélioration de la durée de vie.
À l'inverse, certaines variables ont un effet négatif sur l'espérance de vie, ce qui pourrait
indiquer des domaines nécessitant des interventions spécifiques pour améliorer la santé
publique.
Le modèle permet de faire des prédictions fiables sur l'espérance de vie en fonction des
variables indépendantes considérées. Toutefois, d'autres variables et des modèles plus
complexes pourraient être explorés pour améliorer la précision des prédictions et mieux
comprendre les différents facteurs affectant l'espérance de vie


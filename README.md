# Prédiction de l'espérance de vie à l'aide de la régression linéaire simple

## 1) Exploration des données :
- Pour explorer les données on utilise : ```df = d.read_csv('Life_Expectancy_Data.csv')```
- Pour tracer l’histogramme on utilise : ```sns.histplot(df['Life_expectancy'], kde=True)```
![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/ab6c2007-740e-47a3-8ea3-2bca1f5eddfe)
### Pays en développement contre pays développés
![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/07242b54-1d8d-4c84-8d2f-b4b94e467da2)
Dans l’ensemble de données, il y a plus de pays en développement que de pays développés

## 2) Traitement des valeurs aberrantes :
<div style="display:flex;">
  ![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/180b6d2b-f13e-4d40-b817-510718eddb8f)
  ![image](https://github.com/Medamine-Bahassou/Prediction-de-l-esperance-de-vie-a-l-aide-de-la-regression-lineaire-simple/assets/146652318/0d36e8ec-294a-47cf-9ac4-94b6019815ae)
</div>

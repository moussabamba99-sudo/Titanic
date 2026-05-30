import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#...............................
# CHARGEMENT DES DONNEES
#...............................

data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(data_url)
print (df.head())
print (df.info())
print (df["Survived"].value_counts())
print (df.groupby("Sex")["Survived"].mean())

#......................................................
# CREATION DE NOUVELLES VARIABLES (feature engineering)
#......................................................

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["Title"] = df["Name"].str.extract(r' ([A-za-z]+)\.', expand=False)
df["Title"] = df["Title"].replace([
    "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
df["Has_Cabin"] = df["Cabin"].notnull().astype(int)
df = df.drop(columns=["Cabin"])
df["Sex"] = df["Sex"].map({"female": 1, "male": 0})

df =df[["Survived", "Pclass", "Sex", "Age", "FamilySize", "Title", "Has_Cabin", "Fare", "Embarked"]]

#...................................................................................................
# 1. Definition des colonnes (Feature groups)
#...................................................................................................
x = df[["Pclass", "Age", "FamilySize", "Title", "Has_Cabin", "Sex", "Fare"]]     
y = df["Survived"]

numeric_features = ["Pclass", "Age", "FamilySize", "Has_Cabin", "Sex", "Fare"]        #Colonnes numeriques
categorical_features = ["Title"]                                                      #Colonnes categorielles
#...................................................................................................
# PIPELINE DE TRANSFORMATIONS (Transformation pipelines)
#...................................................................................................
# Pipeline pour variables numeriques

numeric_pipeline = Pipeline(steps=[
("imputer", SimpleImputer(strategy="median")),     # Imputation des valeurs manquantes
("scaler", StandardScaler())                        # Normalisation (Moyenne =0, ecartype=1)
])
#..................................................................................................
# Pipeline pour variables categorielles
#..................................................................................................

categorical_pipeline = Pipeline(steps=[
("imputer", SimpleImputer(strategy="most_frequent")),                                      # Imputaion par mode
 ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False))   # Encodage One-hot
])
#..................................................................................................
# Preprocesseur (preprocessor / Column transformer)
#..................................................................................................
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),      # Appliquer pipeline numerique
    ("cat", categorical_pipeline, categorical_features)  # Appliquer pipeline categorielle
])
#.................................................................................................
# PIPELINE COMPLET (Full Ml Pipeline)
#.................................................................................................

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),   #Etape 1: pretraitement
    ("classifier", XGBClassifier(      # Etape 2: modele de classification
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])
#.....................................................................................................
# ENTRAINEMENT ET PREDICTION
#....................................................................................................

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

pipeline.fit(x_train, y_train) # entraienemnt du modele

y_pred = pipeline.predict(x_test)  # prediction
print (classification_report(y_test, y_pred, digits=4))

#...........................................................
#EVALUATION PAR CROISEMENT DE VALIDATION (Cross-Validation)
#...........................................................

scores = cross_val_score(
    pipeline,                #Pipeline complet a evaluer
    x,                       # Features d'entrainement (x_train)
    y,                        #Variable cible (y_train)
    cv=5                      # Nombre de folds (iteration de validation)
)
print (scores)                       # Afficher les scores par fold
print ("Moyenne :", scores.mean())   # score moyen
print ("Ecartype: ", scores.std())   # Afficher la varabilite des performances
#......................................................
# EXTRACTION DES NOMS DES FEATURES APRES PRETRAITEMENT
#......................................................
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

#..................................................
# EXTRACTION DES IMPORTANCES DES FEATURES DU MODELE
#..................................................

importances = pipeline.named_steps["classifier"].feature_importances_

#.......................................
# CREATION D'UN DATAFRAME POUR L'ANALYSE
#.......................................

importance_df = pd.DataFrame({
    "feature": feature_names,             # Colonne avec les noms
    "importance": importances              # Colonnes avec les importances
}).sort_values(by="importance", ascending=False) # Afficher par ordre d'importances
print (importance_df)

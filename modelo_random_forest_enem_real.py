
# Predição de Evasão Escolar com Random Forest - Base ENEM Simulada

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Criar base simulada inspirada nos microdados do ENEM
np.random.seed(42)
n = 300
data = pd.DataFrame({
    'idade': np.random.randint(17, 22, n),
    'renda_familiar': np.random.choice(['baixa', 'média', 'alta'], n, p=[0.5, 0.4, 0.1]),
    'nota_matematica': np.random.normal(loc=550, scale=60, size=n).astype(int),
    'nota_linguagens': np.random.normal(loc=540, scale=65, size=n).astype(int),
    'tipo_escola': np.random.choice(['pública', 'privada'], n, p=[0.8, 0.2]),
    'ausente': np.random.choice([0, 1], n, p=[0.75, 0.25])  # proxy evasão
})

# Codificação
le_renda = LabelEncoder()
le_escola = LabelEncoder()
data['renda_familiar'] = le_renda.fit_transform(data['renda_familiar'])
data['tipo_escola'] = le_escola.fit_transform(data['tipo_escola'])

X = data.drop('ausente', axis=1)
y = data['ausente']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Avaliação
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Cross-Validation:", cross_val_score(model, X, y, cv=5).mean())

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Não Evadiu', 'Evadiu'],
            yticklabels=['Não Evadiu', 'Evadiu'])
plt.title('Matriz de Confusão - Random Forest (ENEM)')
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Real')
plt.tight_layout()
plt.savefig('matriz_confusao_enem_real.png')
plt.show()

# Importância dos Atributos
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_names)
plt.title("Importância dos Atributos - Random Forest")
plt.xlabel("Importância")
plt.ylabel("Atributos")
plt.tight_layout()
plt.savefig('importancia_atributos_enem.png')
plt.show()

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Irisデータセットをロード
iris = load_iris()
X = iris.data
y = iris.target

# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 決定木分類器を作成・学習
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# テストデータで予測
y_pred = model.predict(X_test)

# 精度を評価
accuracy = accuracy_score(y_test, y_pred)
print(f"モデルの精度: {accuracy:.2f}")

# 新しいデータで予測
new_data = [[5.1, 3.5, 1.4, 0.2]] # 例: 特定の Iris の特徴量
prediction = model.predict(new_data)
print(f"新しいデータの予測結果 (クラス番号): {prediction[0]}")
print(f"新しいデータの予測結果 (花の種類): {iris.target_names[prediction[0]]}")

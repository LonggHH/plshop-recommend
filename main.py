import json
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify
from flask_cors import CORS

# Kết nối đến PostgreSQL và lấy dữ liệu
try:
    connection = psycopg2.connect(
        dbname='pl_store',
        user='postgres',
        password='postgres',
        host='localhost',
        port=5432
    )
    
    df = pd.read_sql_query("SELECT id, name, price, specifications FROM products;", connection)
    connection.close()
except psycopg2.Error as e:
    print(f"Lỗi kết nối đến PostgreSQL: {e}")

# Xử lý dữ liệu specifications
df['specifications'] = df['specifications'].str.lower().str.replace('[^\w\s]', '', regex=True)
vectorizer = TfidfVectorizer(stop_words='english', max_features=48)
tfidf_matrix = vectorizer.fit_transform(df['specifications'])

# Giảm chiều dữ liệu bằng PCA
pca = PCA(n_components=10)
tfidf_matrix_pca = pca.fit_transform(tfidf_matrix.toarray())

# Chuẩn hóa price
scaler = StandardScaler()
df['price_scaled'] = scaler.fit_transform(df[['price']])

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix_pca, df['price_scaled'], test_size=0.2, random_state=42)

# Hyperparameter tuning với GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Đánh giá mô hình
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Optimized Mean Squared Error: {mse}")
print(f"Optimized Mean Absolute Error: {mae}")

# Flask API
app = Flask(__name__)
CORS(app)

@app.route('/recommend/<int:product_id>', methods=['GET'])
def recommend(product_id):
    try:
        # Lấy dữ liệu sản phẩm đầu vào
        input_data = df[df['id'] == product_id]
        if input_data.empty:
            return jsonify({
                'statusCode': 404,
                'message': 'Product not found',
                'data': None
            })

        # Chuẩn bị dữ liệu cho dự đoán
        input_tfidf = vectorizer.transform(input_data['specifications']).toarray()
        input_pca = pca.transform(input_tfidf)

        # Dự đoán giá
        lstm_prediction_scaled = best_model.predict(input_pca)
        lstm_prediction = scaler.inverse_transform(lstm_prediction_scaled.reshape(-1, 1)).flatten()

        # Tính toán độ tương đồng cosine
        cosine_sim = cosine_similarity(tfidf_matrix_pca, input_pca)
        df['cosine_similarity'] = cosine_sim.flatten()

        # Tính điểm cuối cùng
        df['final_score'] = 0.5 * lstm_prediction[0] + 0.5 * df['cosine_similarity']

        # Đề xuất sản phẩm
        recommended_products = df.sort_values(by='final_score', ascending=False).head(5)
        response_data = {
            'statusCode': 200,
            'message': 'Success',
            'data': recommended_products[['id', 'name', 'price']].to_dict('records')
        }
        return jsonify(response_data)
    except Exception as e:
        response_data = {
            'statusCode': 500,
            'message': str(e),
            'data': None
        }
        return jsonify(response_data)

# Run server
if __name__ == '__main__':
    app.run(port=3006, debug=True)

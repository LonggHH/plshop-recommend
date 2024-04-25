import keras
import psycopg2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Kết nối đến PostgreSQL
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
# X_train và X_test: chứa dữ liệu đặc trưng (feature data) của tập huấn luyện và tập kiểm tra tương ứng.
# y_train và y_test: chứa giá trị mục tiêu (target values) của tập huấn luyện và tập kiểm tra tương ứng.
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix.toarray(), df['price_scaled'], test_size=0.2, random_state=42)

# Xây dựng mô hình LSTM
# Số lượng cột (hoặc đặc trưng) trong ma trận TF-IDF
input_shape = (tfidf_matrix.shape[1],)
model = tf.keras.Sequential([
    keras.layers.Embedding(input_dim=len(vectorizer.vocabulary_), output_dim=50, input_length=input_shape[0]),
    keras.layers.LSTM(50, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(50),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Huấn luyện mô hình với Early Stopping và sẽ ngừng khi mô hình khi bị overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Đánh giá mô hình
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"Mean Absolute Error: {mae}")

@app.route('/recommend/<int:product_id>', methods=['GET'])
def recommend(product_id):
    try:
        # Dự đoán giá và tính toán độ tương đồng cosine
        input_data = df[df['id'] == product_id][['price', 'specifications']]
        input_tfidf = vectorizer.transform(input_data['specifications']).toarray()

        lstm_prediction_scaled = model.predict(input_tfidf)
        lstm_prediction = scaler.inverse_transform(lstm_prediction_scaled)

        cosine_sim = cosine_similarity(tfidf_matrix, input_tfidf)
        df['cosine_similarity'] = cosine_sim.flatten()

        # Tính điểm cuối cùng
        df['final_score'] = 0.5 * lstm_prediction.flatten() + 0.5 * df['cosine_similarity']

        # Đề xuất sản phẩm
        recommended_products = df.sort_values(by='final_score', ascending=False).head(5)
        # print(recommended_products[['id', 'name', 'price', 'specifications', 'final_score']])
        response_data = {
            'statusCode': 200,
            'message': 'Success',
            # 'data': recommended_products[['id', 'name', 'price', 'specifications', 'final_score']].to_dict('records')
            'data': recommended_products[['id', 'name']].to_dict('records')
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

# Lưu mô hình
# model.save('product_recommendation_model.h5')
# keras.models.save_model(model=model, filepath='product_recommendation_model.h5', overwrite=True)

# Vẽ biểu đồ lịch sử mô hình
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss Over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

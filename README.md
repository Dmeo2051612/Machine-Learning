# Phân Cụm Các Quốc Gia Theo Chỉ Số Hạnh Phúc Thế Giới (2019) Bằng Thuật Toán DBSCAN

Dự án áp dụng thuật toán phân cụm dựa trên mật độ **DBSCAN** (cài đặt từ đầu bằng Python) trên bộ dữ liệu **World Happiness Report 2019 (`2019.csv`)** nhằm nhóm các quốc gia có đặc điểm kinh tế - xã hội tương đồng và tự động phát hiện các quốc gia có cấu trúc chỉ số cá biệt (ngoại lai tự nhiên).

---

## 1. Tổng Quan Bộ Dữ Liệu

* **Nguồn dữ liệu:** World Happiness Report 2019 (`2019.csv`)
* **Kích thước:** `156` dòng (tương ứng 156 quốc gia/vùng lãnh thổ) $\times$ `9` cột.
* **Chất lượng dữ liệu thô:** Không có giá trị bị khuyết (`0` NaN) và không có bản ghi trùng lặp (`0` duplicates).

| Tên cột | Kiểu dữ liệu | Vai trò trong bài toán | Mô tả ý nghĩa |
| :--- | :---: | :---: | :--- |
| `Overall rank` | `int64` | Loại bỏ | Thứ hạng hạnh phúc của quốc gia ($1 \rightarrow 156$) |
| `Country or region` | `object` | Giữ riêng làm nhãn | Tên quốc gia/vùng lãnh thổ (dùng để đối chiếu sau khi gom cụm) |
| `Score` | `float64` | Loại bỏ | Tổng điểm hạnh phúc (thang điểm Cantril ladder) |
| `GDP per capita` | `float64` | **Đặc trưng (Feature 1)** | Mức độ đóng góp của GDP bình quân đầu người |
| `Social support` | `float64` | **Đặc trưng (Feature 2)** | Mức độ hỗ trợ xã hội |
| `Healthy life expectancy` | `float64` | **Đặc trưng (Feature 3)** | Tuổi thọ khỏe mạnh kỳ vọng |
| `Freedom to make life choices` | `float64` | **Đặc trưng (Feature 4)** | Quyền tự do lựa chọn cuộc sống |
| `Generosity` | `float64` | **Đặc trưng (Feature 5)** | Mức độ hào phóng, hoạt động từ thiện |
| `Perceptions of corruption` | `float64` | **Đặc trưng (Feature 6)** | Nhận thức/niềm tin về mức độ minh bạch, chống tham nhũng |

---

## 2. Quy Trình Tiền Xử Lý Dữ Liệu (Data Preprocessing)

Chất lượng phân cụm của DBSCAN phụ thuộc trực tiếp vào cách định nghĩa không gian đặc trưng và thước đo khoảng cách Euclidean. Quy trình tiền xử lý được thực hiện qua **4 bước** chặt chẽ:

### Bước 1: Kiểm tra dữ liệu khuyết thiếu và giá trị `0.0`
* Kiểm tra toàn bộ bảng dữ liệu xác nhận không tồn tại giá trị `NaN` hay dòng trùng lặp.
* Một số cột có chứa giá trị `0.0` tại quốc gia đứng cuối tiêu chí đó (ví dụ: *Somalia* ở `GDP per capita`, *Central African Republic* ở `Social support`, *Afghanistan* ở `Freedom to make life choices`). Đây là điểm quy đổi thấp nhất phản ánh đúng thực tế thống kê chứ không phải lỗi thiếu dữ liệu, đồng thời không gây lỗi toán học trong phép tính khoảng cách Euclidean nên được giữ nguyên.

### Bước 2: Lựa chọn đặc trưng & Loại bỏ cột định danh / rò rỉ dữ liệu
* **Tách cột định danh (`Country or region`):** Lưu riêng vào biến `ten_quoc_gia`. Không đưa cột này vào tập huấn luyện vì dữ liệu dạng chuỗi ký tự không tính được khoảng cách hình học; sau khi DBSCAN trả về mảng nhãn cụm, biến này dùng để ánh xạ ngược lại tên từng quốc gia.
* **Loại bỏ `Overall rank` và `Score`:** `Overall rank` là thứ hạng được suy ra trực tiếp từ `Score` (gây rò rỉ dữ liệu), còn `Score` là điểm tổng hợp của các yếu tố thành phần. Giữ lại **6 cột đặc trưng độc lập** (`dt_new` có kích thước `156 x 6`) giúp mô hình phân cụm thuần túy dựa trên cấu trúc kinh tế - xã hội của từng nước.

### Bước 3: Chuẩn hóa thang đo bằng Z-score (`StandardScaler`)
* **Vấn đề:** Mặc dù cả 6 đặc trưng đều được quy đổi theo thang điểm đóng góp vào chỉ số hạnh phúc, độ biến thiên (phương sai) giữa các cột chênh lệch rất lớn. Cụ thể, `GDP per capita` trải dài từ `0.000` đến `1.684`, trong khi `Perceptions of corruption` chỉ dao động từ `0.000` đến `0.453`. Nếu không chuẩn hóa, các cột có biên độ lớn sẽ chi phối hoàn toàn khoảng cách Euclidean và làm vô hiệu hóa các cột có biên độ nhỏ.
* **Giải pháp:** Áp dụng chuẩn hóa Z-score (`StandardScaler`) cho toàn bộ 6 đặc trưng theo công thức:

$$z = \frac{x - \mu}{\sigma}$$

* **Kết quả:** Đưa cả 6 cột về cùng một mặt bằng thang đo với trung bình $\mu = 0$ và độ lệch chuẩn $\sigma = 1$, đảm bảo **6 yếu tố xã hội có trọng số công bằng ngang nhau** khi tính khoảng cách.

### Bước 4: Bảo toàn các điểm ngoại lai tự nhiên (Natural Outliers)
* **Quyết định kỹ thuật:** Không sử dụng ngưỡng cắt (`clip`) hay xóa bỏ các quốc gia có giá trị Z-score vượt khoảng $[-3, 3]$ (như *Myanmar*, *Indonesia* có chỉ số `Generosity` cao đột biến; *Singapore*, *Rwanda* có `Perceptions of corruption` cao vượt trội; hay *Central African Republic* có `Social support` rất thấp).
* **Lý do:**
  1. Dữ liệu trong World Happiness Report là số liệu thống kê thực tế đã kiểm duyệt, phản ánh đặc thù văn hóa - xã hội riêng biệt của từng quốc gia (*Natural Outliers*), không phải nhiễu kỹ thuật do nhập liệu sai.
  2. Điểm mạnh cốt lõi của **DBSCAN** là khả năng tự động nhận diện các điểm mật độ thấp và tách chúng thành nhãn nhiễu (`-1`). Việc giữ nguyên dữ liệu giúp thuật toán phát hiện chính xác các quốc gia cá biệt này thay vì làm biến dạng số liệu gốc.

---

## 3. Lựa Chọn Tham Số & Cài Đặt Thuật Toán DBSCAN

### 3.1. Xác định tham số `min_pts` và `eps`
* **Chọn `min_pts = 7`:** Áp dụng quy tắc kinh nghiệm cho dữ liệu đa chiều $\text{minPts} \ge D + 1$. Với số chiều đặc trưng $D = 6$, chọn $\text{min\_pts} = 6 + 1 = 7$.
* **Chọn `eps = 1.8` qua biểu đồ K-distance:**
  1. Sử dụng `NearestNeighbors(n_neighbors=7)` để tính khoảng cách Euclidean từ mỗi quốc gia tới láng giềng thứ `7`.
  2. Sắp xếp các khoảng cách này theo thứ tự tăng dần và vẽ đồ thị **K-distance Graph**.
  3. Quan sát điểm "khuỷu tay" (Elbow) nơi đường cong bắt đầu gãy khúc và tăng dốc mạnh để chốt bán kính quét **`eps = 1.8`**.

### 3.2. Cài đặt thuật toán DBSCAN từ đầu (From Scratch)
Thuật toán được xây dựng thủ công theo đúng nguyên lý phân cụm dựa trên mật độ thông qua các hàm chức năng riêng biệt:
* `euclidean(diem_a, diem_b)`: Tính khoảng cách Euclidean giữa 2 quốc gia trong không gian 6 chiều.
* `tim_neighbor(diem, du_lieu, eps)`: Truy vấn tất cả các quốc gia nằm trong vùng lân cận bán kính `eps` của một điểm.
* `check_core_point(diem, du_lieu, eps, min_pts)`: Kiểm tra điều kiện điểm lõi (Core Point) khi số lượng láng giềng $\ge \text{min\_pts}$.
* `mo_rong_clusters(...)`: Lan truyền mật độ (Density-Reachable) từ điểm lõi sang các láng giềng đạt mật độ để mở rộng cụm.
* `dbscan(du_lieu, eps, min_pts)`: Duyệt toàn bộ tập dữ liệu, gán nhãn cụm (`0, 1, ...`) cho các vùng đạt mật độ và gán nhãn `-1` cho các điểm nhiễu.

---

## 4. Kết Quả Thực Nghiệm & Trực Quan Hóa

### 4.1. Kết quả phân cụm (`eps = 1.8`, `min_pts = 7`)
* **Số cụm tìm được:** `1` cụm chính (**Cụm 0**).
* **Phân bổ dữ liệu:**
  * **Cụm 0 (`149` quốc gia — chiếm `95.5%`):** Cho thấy trên không gian 6 chỉ số kinh tế - xã hội, phần lớn các quốc gia trên thế giới phân bố thành một dải mật độ liên tục chuyển tiếp dần từ nhóm đang phát triển lên nhóm phát triển cao.
  * **Nhóm nhiễu / Ngoại lai tự nhiên (`7` quốc gia — nhãn `-1`):**
    1. `Uzbekistan`
    2. `Indonesia`
    3. `Somalia`
    4. `Myanmar`
    5. `Syria`
    6. `Rwanda`
    7. `Central African Republic`
  * *Nhận xét:* Danh sách 7 quốc gia bị DBSCAN cô lập vào nhóm nhiễu hoàn toàn trùng khớp với các quốc gia có chỉ số cá biệt vượt trội hoặc tụt sâu đã phân tích ở bước tiền xử lý.

### 4.2. Trực quan hóa kết quả bằng PCA (2D)
* Sử dụng **PCA (`n_components=2`)** để giảm chiều dữ liệu từ 6 chiều xuống mặt phẳng 2 chiều (bảo toàn **`73.59%`** tổng phương sai của dữ liệu gốc).
* Vẽ biểu đồ phân tán (`Scatter Plot`) biểu diễn trực quan vùng phân bố dày đặc của **Cụm 0** và vị trí tách biệt của **7 điểm nhiễu (`x`)** ở vùng rìa không gian.

---

## 5. Yêu Cầu Môi Trường & Hướng Dẫn Chạy

### Thư viện sử dụng
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Các bước thực thi
1. Mở file Jupyter Notebook (`.ipynb`) trong môi trường VS Code hoặc Jupyter Lab.
2. Chạy tuần tự các cell từ trên xuống dưới:
   * Đọc dữ liệu trực tiếp từ đường dẫn GitHub raw (`2019.csv`).
   * Tiền xử lý chọn 6 đặc trưng và chuẩn hóa bằng `StandardScaler`.
   * Vẽ biểu đồ `K-distance Graph` kiểm tra điểm khuỷu tay `eps = 1.8`.
   * Chạy hàm `dbscan(dt_chuan_hoa, eps=1.8, min_pts=7)` để lấy nhãn cụm.
   * Giảm chiều bằng `PCA(n_components=2)`, vẽ biểu đồ phân cụm và xuất danh sách quốc gia theo từng nhóm.
# Đồ án môn học Lập trình song song ứng dụng

## 1. Danh sách thành viên
|MSSV|Họ tên|
|---|---|
|19120364|Nguyễn Đắc Thắng (Nhóm trưởng)|
|19120186|Đỗ Lê Khánh Đăng|
|19120462|Lục Minh Bửu|

## 2. Kế hoạch hàng tuần

|Tuần|Công việc|
|---|---|
|1 (từ ngày 20/3 đến ngày 26/3)|- Viết báo cáo đề tài. <br> - Tìm hiểu về thuật toán HOG và SVM.|
|2 (từ ngày 27/3 đến ngày 2/4)|- Tìm hiểu thuật toán SVM. <br> - Thiết kế kiến trúc cho thuật toán HOG phiên bản tuần tự.|

## 3. Giới thiệu đồ án

Trong đồ án này, nhóm sử dụng hai thuật toán là Histogram of Oriented Gradients (HOG) và Support Vector Machine (SVM) để thực hiện bài toán phân loại ảnh.

Thuật toán HOG dùng để trích xuất đặc trưng của một bức ảnh. Output của thuật toán này sẽ là một vector đặc trưng làm input của thuật toán SVM.

Ưu điểm: tốc độ nhanh hơn, tốn ít tài nguyên hơn, kết quả có thể tốt hơn với các bộ dữ liệu nhỏ.

### a. Thuật toán HOG

Bước 1: Tính gradient.

Bước 2: Tính mức độ thay đổi (magnitude) và hướng thay đổi (direction) của Gradient.

Bước 3: Tìm histogram của gradient cell.

Bước 4: Chuẩn hóa block.

Bước 5: Tính vector đặc trưng HOG.

&rarr; Đây là công việc tiêu tốn nhiều tài nguyên tính toán và có thể song song hóa.

### b. Thuật toán SVM

![Hình ảnh giới thiệu thuật toán SVM](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiRduiIq9xoknDqzSp6dqNPyWOH5qJs43DLl-aAfMFlCvl9ye7c5t33TimmeKdAjuOlXhCSi7V_eSXwGx--HdcE4RbvXgLzLYvPMqTvUYGJ00RWeVwHk0wLCOa0F6bvTVWXvg0LAC5U3t5D75z1J4EZ3r9R95eEBv3HufyicLMlxrcG0qXd_oA3Z7tzpA/s668/svm.png)

[Nguồn ảnh](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiRduiIq9xoknDqzSp6dqNPyWOH5qJs43DLl-aAfMFlCvl9ye7c5t33TimmeKdAjuOlXhCSi7V_eSXwGx--HdcE4RbvXgLzLYvPMqTvUYGJ00RWeVwHk0wLCOa0F6bvTVWXvg0LAC5U3t5D75z1J4EZ3r9R95eEBv3HufyicLMlxrcG0qXd_oA3Z7tzpA/s668/svm.png)

Giải thích ý nghĩa các từ trong hình:

`Decision boundary`: đường ra quyết định, khi có một giá trị mới thì đường này dùng để quyết định xem giá trị này thuộc về lớp nào.

`Positive / Negative hyperplane`: đường này được dùng khi huấn luyện dữ liệu, cho phép giá trị ngoại lai (outliers) được học sai bởi thuật toán --> tránh việc bị thiên vị (bias) bởi mô hình.

`Maximum marginal distance`: khoảng cách dài nhất giữa decision boundary và support vectors.

`Support vectors`: những giá trị mà nằm trong khoảng từ negative hyperplane đến positive hyperplane.

## 4. Mục tiều đồ án

Mức 100%: hoàn thành được thuật toán HOG phiên bản tuần tự, song song và các phiên bản cải tiến với độ chính xác cao và tối ưu thời gian thực thi, có thể áp dụng vào các bài toán real-time.

Mức 125%: hoàn thành toàn bộ hệ thống nhận diện gồm HOG + SVM ở cả phiên bản tuần tự, song song và cải tiến.

Mức 75%: cài đặt được HOG tuần tự và song song.

## 5. Tài nguyên và thách thức

### Tài nguyên

Thiết bị, phần cứng, môi trường chạy: Laptop cá nhân, Google Colab.

Các công cụ hỗ trợ: Github, Colab.

### Thách thức

Thuật toán HOG là một thuật toán chậm nên cần cải tiến nhiều.

Kết hợp cả 2 model HOG và SVM.

Cả hai thuật toán chạy song song hiệu quả hơn.


1. Thông tin thành viên
19120186 - Đỗ Lê Khánh Đăng
19120364 - Nguyễn Đắc Thắng
19120462 - Lục Minh Bửu

2. Bài làm của nhóm sẽ có 3 phần chính:
- 2 file HOG_detail và SVM_detail:
    + Trình bày chi tiết về thuật toán
    + Trình bày ý tưởng và các bước song song hóa
    + Link tham khảo của các thuật toán.
- Thư mục demo và so sánh:
    + Gồm 3 file .py là 3 file chứa các class và các hàm cần dùng khi chạy thuật toán,
    mục tiêu là để notebook so sánh đẹp hơn.
    + Các file ?_vs_?.ipynb là các file thực hiện so sánh và đánh giá các so sánh đó
    + demo.ipynb: file cho phép thử chạy thuật toán đã học sẵn để nhận diện 1 ảnh màu bất kỳ
- slide:
    + Chỉnh sửa và bổ sung từ slide Seminar 3 của nhóm
    + Là file để báo cáo nhận xét các ý chung của nhóm.

- Link kaggle dữ liệu Cat và Dog: https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/code?fbclid=IwAR3SBq-BbUfVVLdFrD-oz-5zuU20rcaVE94RkiZBxQXgTvQRUuMhdRSkZp8

3. Một số thay đổi:
- Nhóm đã fix warning về truyền host array vào kernel function
- Với warning về gridsize không thỏa occupancy:
    + Do số lượng block dùng không đủ nhiều để tất cả SM đều hoạt động nên sẽ hiện thông báo.
    + Với dữ liệu ít (như bộ test tầm vài trăm ảnh) sẽ hiện lên thông báo này. Với dữ liệu tầm vài nghìn ảnh thì sẽ không bị
    + Nhóm có thể giảm block size xuống để xử lý thông báo khi dữ liệu nhỏ. Tuy nhiên nó sẽ ảnh hưởng khi dữ liệu lớn
    + Nhóm vẫn chưa tìm được cách để tối ưu blocksize với mọi kích thước dữ liệu

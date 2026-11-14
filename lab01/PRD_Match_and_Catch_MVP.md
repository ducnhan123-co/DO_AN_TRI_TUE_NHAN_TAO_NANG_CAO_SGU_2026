# PRD — Match & Catch: Flying Cloud Services (MVP 2–3 ngày)

| Thuộc tính | Nội dung |
|---|---|
| Sản phẩm | Web minigame tại sự kiện CLB, chơi trên điện thoại bằng cách quét QR |
| Chủ đề | Nhận diện dịch vụ AWS và Google Cloud; nhận biết chức năng dịch vụ qua câu bonus |
| Đối tượng | Sinh viên tham gia và ban tổ chức |
| Phiên bản | MVP 0.3 — 21/09/2026 |
| Cách chơi tham khảo | [Wordwall — Flying Fruit](https://wordwall.net/resource/67301005/flying-fruit) |

## 1. Mục tiêu và giới hạn phạm vi

Người chơi **quét một mã QR chung** để mở website trên điện thoại, nhập tên và mã sinh viên, rồi chạm vào các thẻ tên dịch vụ đang bay thuộc nhà cung cấp mục tiêu (**AWS hoặc Google Cloud**). Khi chọn đúng **5 thẻ chính liên tiếp**, người chơi mở khóa **1 câu bonus** về chức năng dịch vụ. Mỗi lượt có **tối đa 2 câu bonus**. Web lưu kết quả và hiển thị bảng xếp hạng để xác định TOP 3.

**Bắt buộc trong MVP:** QR dẫn đến URL game; giao diện điện thoại; nhập tên và mã sinh viên; 2 vòng chơi chính; 3 tim; tối đa 2 bonus; điểm và màn kết quả; lưu kết quả; bảng xếp hạng; thông báo đủ điều kiện quà.

**Không làm trong MVP:** app Android/iOS; giao diện chơi riêng cho máy tính; đăng nhập/tài khoản/mật khẩu người chơi; đăng nhập trường; trang admin; quản trị ngân hàng câu hỏi qua UI; mã nhận quà tự động; hệ thống xác nhận phát quà; âm thanh; hiệu ứng phức tạp; WebSocket/realtime; đối kháng; kéo-thả; phân loại “Cả 2”; ERD trong giai đoạn này.

**Phát quà:** ban tổ chức đối chiếu mã sinh viên và danh sách kết quả **thủ công**, chỉ phát quà thường một lần cho mỗi mã sinh viên.

## 2. Luồng người chơi

1. Quét QR chung tại booth → mở URL website trên điện thoại.
2. Nhập **tên hiển thị + mã sinh viên** → đọc hướng dẫn ngắn → bấm **Bắt đầu**.
3. Chơi 2 vòng; mỗi vòng có **một** nhà cung cấp mục tiêu, AWS hoặc Google Cloud. Hai vòng xuất hiện theo thứ tự ngẫu nhiên.
4. Chạm thẻ đúng để lấy điểm. Khi đạt 5 lựa chọn đúng liên tiếp và chưa dùng hết 2 bonus, game tạm dừng để hiển thị câu bonus. Hoàn thành bonus rồi tiếp tục màn chính.
5. Khi hoàn thành hai vòng hoặc mất hết 3 tim, xem **điểm chính + điểm bonus + tổng điểm**, thông báo điều kiện quà và bảng xếp hạng.
6. Mỗi mã sinh viên được chơi tối đa **3 lượt**; chỉ lượt có thành tích cao nhất của mỗi người xuất hiện trên bảng.

Không yêu cầu tạo tài khoản; **mã sinh viên chỉ là định danh khai báo, không phải xác thực danh tính**.

## 3. Luật chơi chính

### 3.1. Cấu trúc và thẻ bay

| Thông số | Quy định MVP |
|---|---|
| Số vòng | 2: một vòng AWS, một vòng Google Cloud |
| Mỗi vòng | 10 thẻ dịch vụ mục tiêu + 10 thẻ gây nhiễu = 20 thẻ |
| Cả lượt | 20 thẻ mục tiêu + 20 thẻ gây nhiễu |
| Tim | 3 tim dùng chung hai vòng, không hồi tim |
| Thẻ cùng xuất hiện | **Tối đa 2**, tránh che nhau trên màn hình điện thoại |
| Tương tác | **Chạm một lần** vào thẻ, không kéo-thả |
| Thời gian thẻ | Cấu hình đơn giản trong mã; thử trên điện thoại thật để chốt tốc độ |
| Chuyển vòng | Khi tất cả 20 thẻ của vòng hiện tại được xử lý và vẫn còn tim |

Tên và logo nhà cung cấp mục tiêu, điểm và tim luôn hiển thị rõ phía trên vùng chơi. Các thẻ bay ngang/qua lại trong vùng không che thông tin cố định và không đè lên nhau. Mỗi thẻ chỉ được xử lý **một lần**: chạm xong thì biến mất; không chạm thì được xử lý khi hết thời gian xuất hiện. Không cần mô phỏng vật lý hay animation cầu kỳ.

### 3.2. Điểm, tim và chuỗi đúng

| Tình huống | Điểm chính | Tim | Chuỗi đúng |
|---|---:|---:|---|
| Chạm thẻ thuộc nhà cung cấp mục tiêu | +1 | Không đổi | +1 |
| Chạm thẻ của nhà cung cấp còn lại | 0 | −1 | Về 0 |
| Để thẻ mục tiêu bay mất | 0 | −1 | Về 0 |
| Để thẻ gây nhiễu bay mất | 0 | Không đổi | Không đổi |

- Chuỗi đúng **được giữ khi chuyển vòng**. Chỉ thẻ đúng trong phần chơi chính mới tăng chuỗi; bonus không tăng chuỗi.
- Khi chuỗi đạt **5** và `bonusCount < 2`, **mở ngay 1 bonus**, tăng `bonusCount` thêm 1 và **đặt chuỗi về 0**. Khi đã mở đủ 2 bonus, **không mở thêm** dù tiếp tục chọn đúng.
- Nếu thẻ thứ 5 đúng là thẻ cuối vòng/lượt, xử lý bonus trước rồi mới chuyển vòng/hiện kết quả.
- Khi tim về **0**, kết thúc lượt ngay, hủy thẻ chưa xử lý và không mở bonus mới. Điểm không âm.

## 4. Bonus — hiểu chức năng dịch vụ

**Mục tiêu phân hóa kiến thức:** câu chính hỏi *dịch vụ thuộc nhà cung cấp nào*; bonus hỏi *dịch vụ dùng để làm gì*. Câu bonus phải khó hơn một chút nhưng vẫn là kiến thức cloud nền tảng; không hỏi giới hạn kỹ thuật, mức giá hoặc chi tiết quá chuyên sâu.

**Tần suất:** tối đa **2 câu bonus/lượt**. Mỗi câu được mở sau **một chuỗi 5 thẻ chính đúng liên tiếp**; cần một chuỗi 5 mới để mở câu thứ hai. Không có bonus thứ ba.

**Cách chơi:**

1. Tạm dừng hoàn toàn thẻ chính và bộ đếm thời gian của chúng; không nhận chạm phần chơi chính khi bonus mở.
2. Hiển thị **icon + tên một dịch vụ AWS hoặc Google Cloud** và câu hỏi: “Dịch vụ này chủ yếu dùng để làm gì?”.
3. Hiển thị **3 thẻ mô tả chức năng bay ngang/qua lại**: **1 đáp án đúng, 2 đáp án nhiễu hợp lý**. Người chơi chạm **một** đáp án; không kéo-thả.
4. Chạm đúng: **+3 điểm bonus**. Chạm sai hoặc hết **10 giây** mà chưa chọn: **0 điểm bonus**. **Không trường hợp nào bị trừ tim**, và không có điểm âm.
5. Đóng bonus, tiếp tục thẻ chính đúng tại trạng thái đã tạm dừng; nếu vừa hết vòng/lượt thì chuyển vòng/hiện kết quả.

**Quy tắc kỹ thuật đơn giản:** chỉ một câu bonus hiện tại và chỉ một lựa chọn được ghi nhận. Thẻ đáp án có thể bay lặp trong vùng chơi đến hết 10 giây; **thẻ đáp án bay khỏi mép màn hình không bị tính là bỏ lỡ**. Câu hỏi chọn ngẫu nhiên từ ngân hàng soạn sẵn, **không lặp trong một lượt**. Không sinh câu hỏi bằng AI.

**Độ khó:** bonus 1 dùng chức năng phổ biến, mô tả ngắn, dễ phân biệt; bonus 2 có hai phương án nhiễu gần nhóm chức năng hơn nhưng **chỉ duy nhất một đáp án đúng rõ ràng**. Câu 2 không yêu cầu kiến thức chuyên gia. Ba thẻ đáp án phải đủ ngắn để đọc trên điện thoại; **không tăng tốc độ bay để tạo độ khó**.

## 5. Điểm, quà và bảng xếp hạng

```text
main_score  = số thẻ mục tiêu chọn đúng       (0–20)
bonus_count = số câu bonus đã xuất hiện       (0–2)
bonus_score = số câu bonus trả lời đúng × 3   (0–6)
total_score = main_score + bonus_score        (0–26)
hearts_left = max(0, 3 − wrong − missed)
```

- **Xếp hạng:** tổng điểm cao hơn → nếu hòa, điểm chính cao hơn → nếu vẫn hòa, thời điểm nộp lượt hợp lệ sớm hơn. Nếu một người có nhiều lượt, áp dụng cùng quy tắc để chọn lượt tốt nhất của họ. Hiển thị tên, điểm chính, điểm bonus và tổng điểm; **không công khai mã sinh viên**. TOP 3 là ba người đứng đầu tại thời điểm ban tổ chức chốt sự kiện.
- **Quà thường:** `main_score >= 10` trên 20 điểm chính; **không cộng bonus vào ngưỡng quà**. Hiển thị “Đủ điều kiện nhận quà” ở màn kết quả. Ban tổ chức đối chiếu mã sinh viên và danh sách thủ công trước khi trao.
- **Giới hạn lượt:** tối đa 3 lượt/mã sinh viên; một người chỉ có một vị trí trên bảng. Quà thường trao tối đa một lần/mã sinh viên nhờ đối chiếu thủ công.
- **Cập nhật bảng:** chỉ cần nút **Làm mới** hoặc tải lại trang; không cần cập nhật tự động mỗi vài giây. Ban tổ chức chốt TOP 3 và thông báo ngoài ứng dụng; không cần nút đóng sự kiện.

## 6. Ngân hàng nội dung tĩnh

Lưu một file JSON/TypeScript trong dự án, **không xây hệ thống quản lý nội dung**.

- **Dịch vụ phần chính:** mỗi mục có `id`, `provider` (`aws`/`google`), `name`. Tối thiểu **10 dịch vụ mỗi bên** để lấy đủ 10 thẻ mục tiêu/vòng; nếu có thời gian, thêm vài mục để xáo trộn giữa các lượt.
- **Dữ liệu gợi ý để nhóm rà soát:** AWS — Amazon S3, AWS Lambda, Amazon CloudFront, Amazon EC2, Amazon RDS, Amazon DynamoDB, Amazon SQS, Amazon Route 53, Amazon EKS, Amazon Redshift. Google Cloud — Cloud Storage, Cloud Run, Cloud CDN, Compute Engine, Cloud SQL, Firestore, Pub/Sub, Cloud DNS, Google Kubernetes Engine, BigQuery.
- **Bonus:** cần tối thiểu **2 câu khác nhau** để không lặp trong một lượt; **đề xuất soạn 4–6 câu** để có thể thay đổi nội dung giữa người chơi. Mỗi câu có `serviceId`, `iconPath`, `question`, `options` (3 mô tả ngắn), `correctOptionId`, `difficulty` (`basic`/`intermediate`). Câu bonus 1 lấy nhóm `basic`, câu bonus 2 lấy nhóm `intermediate`; nếu ngân hàng chưa đủ thì dùng câu `basic` khác, không được lặp.
- Icon lưu sẵn trong dự án và sử dụng phù hợp điều kiện sử dụng thương hiệu. Một người phụ trách rà soát tên/icon và **cả ba phương án**, tránh phương án nhiễu cũng đúng trong thực tế.

## 7. Thiết kế điện thoại và triển khai

- **Chỉ yêu cầu giao diện chơi trên điện thoại**, ưu tiên **chiều dọc**, màn hình rộng từ khoảng **360 px**; kiểm thử trình duyệt di động Chrome/Safari. Không làm desktop layout riêng. QR chung trỏ thẳng tới URL web đã triển khai; **không cần hệ thống sinh QR động**.
- Thẻ có vùng chạm đủ lớn (mục tiêu tối thiểu khoảng **44 × 44 px**), không chồng nhau, không che logo/điểm/tim. Tối đa **1–2 thẻ chính đồng thời**. Câu bonus hiển thị câu hỏi rõ và 3 đáp án có thể đọc được; không để các thẻ đáp án che nhau.
- Animation ưu tiên CSS `transform` hoặc cách đơn giản mà nhóm quen thuộc; phải **pause/resume đúng** cho cả chuyển động và thời gian tồn tại của thẻ khi mở bonus. Không tính nhầm `missed` trong lúc bonus.
- Trạng thái tối thiểu: `intro`, `main`, `bonus`, `result`. Chặn xử lý trùng do chạm liên tục hoặc thời điểm chạm trùng timeout/chuyển trạng thái.
- Backend/database tối thiểu lưu mã sinh viên, tên hiển thị, số lượt, `mainScore`, `bonusScore`, `totalScore`, thời điểm nộp; cung cấp API lấy bảng xếp hạng. Dùng công nghệ và hosting **nhóm đã quen**, không dựng thêm hạ tầng phức tạp chỉ vì chủ đề AWS.
- Server áp giới hạn 3 lượt/mã sinh viên, kiểm tra `mainScore` trong 0–20, `bonusScore` thuộc {0, 3, 6}, `totalScore = mainScore + bonusScore` và tổng không vượt 26. Nên ghi nhận `bonusCount` để kiểm tra `bonusScore <= 3 × bonusCount <= 6`. Các kiểm tra giới hạn **không chứng minh được người chơi thật sự trả lời đúng** nếu toàn bộ logic chấm điểm ở trình duyệt; chống gian lận mạnh nằm ngoài MVP.
- Nếu nộp kết quả thất bại, giữ kết quả và có nút **Gửi lại**; dùng mã định danh lượt để lần gửi trùng không trừ thêm lượt.

## 8. Tiêu chí nghiệm thu (Must-have)

| ID | Kiểm tra đạt khi |
|---|---|
| AC-01 | Quét QR bằng điện thoại mở đúng website và có thể nhập tên, mã sinh viên, bắt đầu chơi. |
| AC-02 | Mục tiêu AWS: chạm Amazon S3 +1 điểm chính; chạm dịch vụ Google −1 tim và đặt chuỗi về 0. |
| AC-03 | Bỏ lỡ thẻ mục tiêu −1 tim, đặt chuỗi về 0; để thẻ nhiễu bay mất không phạt và không ngắt chuỗi. |
| AC-04 | Chọn đúng 5 thẻ chính liên tiếp xuất hiện đúng 1 bonus; sau bonus chuỗi về 0 và phải đạt thêm 5 đúng liên tiếp mới có bonus 2. |
| AC-05 | Sau khi đã mở 2 bonus, không xuất hiện bonus thứ 3 dù tiếp tục đúng; hai câu trong cùng lượt không trùng. |
| AC-06 | Bonus có icon + tên dịch vụ và 3 chức năng bay; chọn đúng +3, sai/hết 10 giây +0; không trừ tim. |
| AC-07 | Trong bonus, thẻ chính và thời gian đứng yên, không bị tính bỏ lỡ; đóng bonus tiếp tục bình thường. |
| AC-08 | Nếu thẻ đúng thứ 5 là thẻ cuối vòng/lượt, chơi bonus xong rồi mới chuyển vòng/hiện kết quả. |
| AC-09 | Hết 3 tim thì kết thúc ngay; không có thẻ/bonus mới; một lần chạm/timeout không bị tính hai lần. |
| AC-10 | Tổng điểm = điểm chính + bonus; điểm chính tối đa 20, bonus tối đa 6, tổng tối đa **26**. |
| AC-11 | Đạt từ 10 điểm **chính** thì hiện thông báo đủ điều kiện quà, không phụ thuộc điểm bonus. |
| AC-12 | Mỗi mã sinh viên tối đa 3 lượt; bảng chỉ hiển thị lượt tốt nhất, không lộ mã sinh viên; thứ tự hòa theo mục 5. |
| AC-13 | Chơi, làm bonus, gửi lại kết quả và xem bảng trên màn hình điện thoại khoảng 360 px; gửi lại không tạo lượt trùng. |

## 9. Kế hoạch triển khai 2–3 ngày

| Thời điểm | Công việc ưu tiên | Kết quả tối thiểu |
|---|---|---|
| **Ngày 1** | Nội dung tĩnh; giao diện điện thoại; thẻ bay 1–2 thẻ; chọn đúng/sai/miss, 3 tim, 2 vòng, đếm chuỗi 5 | Chơi trọn phần chính **trên điện thoại** (có thể kiểm thử bằng URL trong mạng nội bộ trước khi deploy). |
| **Ngày 2** | Bonus tối đa 2 câu; 3 đáp án bay + pause/resume + 10 giây; điểm; kết quả; lưu lượt và bảng xếp hạng | Luồng nhập thông tin → chơi → bonus → kết quả → bảng xếp hạng chạy hoàn chỉnh. |
| **Ngày 3 (nếu có)** | Kiểm thử điện thoại thật và mạng di động; kiểm tra câu hỏi, trường hợp cuối vòng và chạm trùng; sửa lỗi; deploy, tạo QR chung | Website hoạt động qua QR và đạt AC-01 đến AC-13. |

**Nếu chỉ có 2 ngày:** bỏ trang trí, hiệu ứng chuyển vòng, thống kê chi tiết số lần sai/bỏ lỡ, bảng xếp hạng dài và tùy chỉnh qua UI. Giữ **lõi game + 2 bonus tối đa + lưu điểm + TOP 3 + URL/QR**. Thông số (tim, số thẻ, thời gian bonus, tốc độ, ngưỡng quà) đặt trong hằng số; không xây trang cấu hình. Không cắt cơ chế tạm dừng thẻ chính hoặc giới hạn lượt vì chúng ảnh hưởng trực tiếp đến kết quả.

## 10. Definition of Done

Người chơi **quét QR và chơi trọn luồng trên điện thoại**; game có 3 tim, 2 vòng và chuỗi 5 đúng; **tối đa 2 bonus** với 3 đáp án bay, đúng +3/sai hoặc hết giờ +0 và không mất tim; **điểm tối đa 26**; điểm lưu được, mỗi mã sinh viên tối đa 3 lượt và TOP 3 hiển thị theo quy tắc; quà xét theo điểm chính; không tính `missed` khi đang ở bonus; câu hỏi đã được rà soát và website hoạt động ở URL trong mã QR.


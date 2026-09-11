# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): 
{
    "class_id": 468,
    "class_name": "cab",
    "rank": 1,
    "score": 0.510915,
    "taxonomy_name": "ImageNet-1K"
  },
- Record này mô tả toàn ảnh như thế nào?
Record này gán nhãn cho ảnh hạng 1 chi tiết trong số 1.000 lớp của taxonomy ImageNet-1K chỉ ra các tham số về id của ảnh, lớp xe taxi (cab) , điểm số của mô hình (model score) dùng để xếp hạng, không phản ánh độ chính xác của nhãn chuẩn (ground truth).

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
Class list do bộ dữ liệu ImageNet-1K định nghĩa từ trước khi huấn luyện mô hình. Mô hình chỉ có thể phân loại trong phạm vi cố định 1.000 lớp này, không thể tự phát sinh hoặc dự đoán lớp mới nằm ngoài taxonomy.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
Cần lưu giữ đủ 3 trường vì:
taxonomy_name xác định hệ quy chiếu chuẩn (ví dụ ImageNet-1K khác với COCO-80 hay OpenImages).
class_id đảm bảo tính duy nhất và không bị nhầm lẫn trong xử lý code/database.
class_name giúp con người đọc hiểu trực quan hơn. Cùng một tên lớp ở các taxonomy khác nhau có thể mang lại định nghĩa và phạm vi khác nhau.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
Guideline cần bổ sung quy tắc ưu tiên rõ ràng khi xử lý ảnh có nhiều đối tượng, tránh tình trạng annotator mỗi người gán nhãn một kiểu khác nhau như là ưu tiên chủ thể có diện tích lớn hơn và nằm ở vị trí trung tâm, hoặc quy định là chỉ chọn ra nhãn tốt nhất hoặc là gắn nhiều nhãn cho các vật thể cùng xuất hiện .
- Vì sao model score không phải ground truth?
Model score chỉ là xác suất toán học (độ tự tin nội bộ của thuật toán) tại thời điểm suy luận. Mô hình hoàn toàn có thể đưa ra score rất cao nhưng vẫn nhận diện sai , hoặc score thấp nhưng lại đoán đúng. Ground truth thì lại là sự thật khách quan: Ground truth là nhãn chuẩn xác thực bởi con người dựa trên guideline và thực tế bức ảnh, chứ không phải con số xác suất của máy móc. Một điều nữa là Model score bị phụ thuộc vào hàm phân phối của riêng checkpoint và taxonomy mà nó được học, trong khi sự vật tồn tại trong bức ảnh là thực thể khách quan không thay đổi."*
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
{
    "class_name": "bus",
    "score": 0.847436,
    "bbox_xyxy": [
      225.61,
      132.68,
      296.09,
      220.45
    ],
    "bbox_width": 70.48,
    "bbox_height": 87.76
  },
- Diễn giải vị trí box bằng lời:
Box này nằm ở tọa đó xmin tại 225.65,ymin tại 132.68,xmax tại điểm 296.09,ymax tại điểm 220.45 có chiều rộng là 70.48 và chiều cao là 87.76
- So sánh số prediction ở hai threshold:
Đối với threshold thấp thì nhiều prediction sẽ có tỉ lệ được giữ lại hơn, kể cả các object nhỏ, bị che khuất hoặc có score thấp. Ở threshold cao thì chỉ các prediction có score cao được giữ lại.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
Độ bao phủ (Recall): Ngưỡng thấp giúp tăng độ bao phủ (ít bị bỏ sót vật thể thật), nhưng tăng tỷ lệ dương tính giả (false positives).
Khối lượng reviewer: Khi hạ threshold để giữ nhiều prediction, khối lượng công việc của reviewer/QC tăng lên đáng kể vì phải duyệt qua nhiều box rác, box sai lệch để xóa bỏ hoặc điều chỉnh.
- Đề xuất một quy tắc box chặt:
Bounding box phải ôm khít các pixel biên ngoài cùng nhìn thấy được của vật thể; không được cắt lẹm vào chi tiết của đối tượng và khoảng trống pixel nền thừa ở 4 cạnh không được vượt quá 2-3 pixel.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
Cần guideline quy định: Nếu vật thể bị che khuất một phần (occluded) nhưng vẫn nhận diện được trên 20% diện tích, annotator sẽ đóng box bao quanh toàn bộ phần nhìn thấy được hay phải ước lượng cả phần bị che.
Cần escalation: Khi vật thể bị che khuất quá nhiều (>80%) hoặc bị cắt mép sát khung hình không thể nhận diện chắc chắn loại vật thể, annotator không tự đoán mà phải gắn cờ (escalate) để xin ý kiến từ Reviewer/Lead.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
{
    "instance_id": "traffic-002",
    "class_name": "car",
    "class_id": 2,
    "score": 0.891092,
    "polygon_point_count": 92,
    "polygon_xy": [
      [
        497.0,
        287.0
      ],
      [
        497.0,
        287.0
      ],
      [
        492.0,
        287.0
      ],
      [
        491.0,
        287.0
      ],
      [
        488.0,
        284.0
      ],
      [
        483.0,
        276.0
      ],
      ...
    ]
  }
- Polygon bổ sung chi tiết gì so với box?
Box chỉ bao quanh object bằng hình chữ nhật bao viền, còn polygon mô tả sát đường biên thật của object bằng rất nhiều điểm tọa độ.
- `instance_id` dùng để làm gì và không phải loại ID nào?
Dùng để phân biệt từng object riêng biệt, kể cả khi chúng cùng lớp.
- Đề xuất một quy tắc biên mask:
Đường biên đa giác phải bám sát đường viền ngoài của vật thể ở mức phóng to tối thiểu 200%; độ lệch biên so với pixel thực không vượt quá 2 pixel; không được để lộ viền pixel nền bên trong đa giác.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Guideline cần làm rõ: Với các vật thể tiếp xúc gần nhau (như chiếc bát đặt chồng lên đĩa), đường ranh giới tiếp xúc sẽ thuộc về đối tượng nào; các chi tiết quá mảnh (dây điện, sợi tóc) có bắt buộc phải vẽ chi tiết hay được phép làm mượt (smooth).
Escalation: Khi hai vật thể cùng màu sắc nằm đè lên nhau ở vùng thiếu sáng khiến mắt thường không thể phân biệt ranh giới pixel, annotator cần escalate để trưởng nhóm quyết định tách hay gộp mask.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Nhãn đơn cấp ảnh (class_id, class_name) | Ảnh có thể chứa đồng thời nhiều chủ thể nhưng chỉ được chọn 1 lớp | Kiểm tra lớp và taxonomy |  Chọn lớp theo guideline, đánh dấu ảnh không rõ|
| Phát hiện vật thể |Bounding box 2D ([x_min, y_min, x_max, y_max] + class_name) |Các nhãn đè lên nhau gây khó quan sát  | Vẽ box riêng cho từng object | Đối chiếu box với ảnh: box có bao trọn vật thể không, có bị thừa quá nhiều khoảng trống nền không, và vật thể có bị che khuất hoặc chạm mép ảnh không.|
| Instance segmentation |Đa giác khép kín (polygon_xy theo pixel + class_name + instance_id) | Các phần nhãn + khung được tô đủ màu sắc và có khả năng đè lên nhãn của vật thể khác gây khó quan sát | Vẽ polygon theo phần object nhìn thấy | Kiểm tra biên mask, vùng chồng lấn và object bị bỏ sót |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
Chỉ sử dụng dữ liệu được phép; không đưa ảnh cá nhân, dữ liệu khách hàng hoặc thông tin nhạy cảm vào notebook, báo cáo hay repository.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
Dừng xử lý và báo cáo lại người phụ tráchgit--vgi

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.

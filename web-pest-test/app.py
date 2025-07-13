from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'E:/Downloads/web_pest/static/uploads'
PLOT_FOLDER = 'E:/Downloads/web_pest/static/plots'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOT_FOLDER'] = PLOT_FOLDER

# التأكد من وجود المجلدات
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# تحميل موديل YOLO
model = YOLO('E:/Downloads/web_pest/bestl.pt')  # تحميل الموديل من ملف weights محلي
model.eval()  # وضع الموديل في وضع التقييم

def plot_bboxes_with_counts(results):
    img = results.orig_img.copy()  # نسخة من الصورة الأصلية
    names = results.names  # قاموس التسميات
    scores = results.boxes.conf.cpu().numpy()  # الاحتمالات (نقلها إلى CPU ثم تحويلها إلى numpy)
    classes = results.boxes.cls.cpu().numpy()  # التصنيفات (نقلها إلى CPU ثم تحويلها إلى numpy)
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)  # المربعات المحيطة

    # حساب أعداد كل تصنيف
    class_counts = Counter(classes)

    # رسم المربعات المحيطة
    for score, cls, bbox in zip(scores, classes, boxes):
        class_label = names[int(cls)]
        label = f"{class_label}: {score:.2f}"
        lbl_margin = 5
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=2)

        label_size = cv2.getTextSize(label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        lbl_w, lbl_h = label_size[0]
        lbl_w += 2 * lbl_margin
        lbl_h += 2 * lbl_margin
        cv2.rectangle(img, (bbox[0], bbox[1] - lbl_h), (bbox[0] + lbl_w, bbox[1]), color=(0, 255, 0), thickness=-1)
        cv2.putText(img, label, (bbox[0] + lbl_margin, bbox[1] - lbl_margin),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)

    return img, class_counts

def save_barplot(class_counts, output_path):
    labels = [f"{int(cls)}" for cls in class_counts.keys()]
    counts = list(class_counts.values())

    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Class Labels')
    plt.ylabel('Counts')
    plt.title('Class Counts')
    plt.savefig(output_path)
    plt.close()

@app.route('/insect-detect')
def index():
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/plots/<filename>')
def plot_file(filename):
    return send_from_directory(app.config['PLOT_FOLDER'], filename)

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'لم يتم رفع صورة'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار صورة'})
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # فتح الصورة باستخدام PIL
        img = Image.open(filepath)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # تحويل الصورة إلى BGR

        # توقع باستخدام YOLO
        results = model(img_cv)
        results = results[0] if isinstance(results, list) else results

        # رسم المربعات وحساب التصنيفات
        img_with_bboxes, class_counts = plot_bboxes_with_counts(results)

        # حفظ الصورة المعدلة
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
        cv2.imwrite(processed_path, img_with_bboxes)

        # حفظ الرسم البياني
        plot_path = os.path.join(app.config['PLOT_FOLDER'], f'plot_{filename}.png')
        save_barplot(class_counts, plot_path)

        # إعداد البيانات للاستجابة
        counts_response = {results.names[int(cls)]: count for cls, count in class_counts.items()}

        return jsonify({
            'original': f'/static/uploads/{filename}',
            'processed': f'/static/uploads/processed_{filename}',
            'plot': f'/static/plots/plot_{filename}.png',
            'class_counts': counts_response
        })

if __name__ == '__main__':
    app.run(debug=True)

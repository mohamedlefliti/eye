# 📦 استيراد المكتبات
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from google.colab import files
from IPython.display import Image, display

# 🎨 إعداد المظهر
sns.set(style='whitegrid')

# 📁 إنشاء مجلد الحفظ
output_path = "/mnt/data"
os.makedirs(output_path, exist_ok=True)

# 📤 رفع صورة الإدخال
print("يرجى رفع صورة (مثل cat.jpg)...")
uploaded = files.upload()
image_name = list(uploaded.keys())[0]

# 📷 تحميل الصورة وتحويلها إلى RGB
img = cv2.imread(image_name)
if img is None:
    raise FileNotFoundError("لم يتم تحميل الصورة بشكل صحيح.")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 🔍 إنشاء مستويات تمويه لمحاكاة العدسة الديناميكية
num_levels = 3
max_blur = 21
blurred_images = []
for i in range(num_levels):
    ksize = 1 + 2 * int((i / (num_levels - 1)) * max_blur // 2)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    blurred_images.append(blurred)

# ✨ تحسين التباين
enhanced = cv2.convertScaleAbs(img, alpha=1.4, beta=25)

# 🔎 إنشاء قناع تركيز دائري (مركز الصورة واضح)
h, w = img.shape[:2]
mask = np.zeros((h, w), dtype=np.uint8)
center = (w // 2, h // 2)
radius = min(h, w) // 3
cv2.circle(mask, center, radius, 255, -1)
mask = cv2.GaussianBlur(mask, (51, 51), 0)
mask_3ch = cv2.merge([mask, mask, mask]) / 255.0

# 🧪 دمج الصورة الأصلية مع آخر مستوى تمويه حسب القناع
sharp_center = enhanced * mask_3ch
blurred_background = blurred_images[-1] * (1 - mask_3ch)
merged = cv2.convertScaleAbs(sharp_center + blurred_background)

# 🧼 إزالة الضوضاء
denoised = cv2.fastNlMeansDenoisingColored(merged, None, 10, 10, 7, 21)

# ⚙️ تصحيح التشوه الهندسي
def undistort_image(image):
    h, w = image.shape[:2]
    K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
    D = np.array([-0.3, 0.1, 0, 0])
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), 5)
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

corrected = undistort_image(denoised)

# 🖼️ عرض وحفظ الصورة النهائية
plt.figure(figsize=(10, 6))
plt.imshow(corrected)
plt.title("Dynamic Lens Simulation Output")
plt.axis('off')
plt.tight_layout()
plt.savefig(f"{output_path}/dynamic_lens_output.png")
plt.close()

# 👁️ عرض الصورة داخل Colab
display(Image(filename=f"{output_path}/dynamic_lens_output.png"))

# 📥 تحميل الصورة إلى الجهاز
files.download(f"{output_path}/dynamic_lens_output.png")

print("✅ تم تنفيذ المحاكاة وحفظ الصورة في:", f"{output_path}/dynamic_lens_output.png")

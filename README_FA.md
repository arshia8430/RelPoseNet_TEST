# پچ مقایسه Lossها برای RelPoseNet

این پچ برای کدی است که کاربر فرستاده است. فایل‌های جایگزین داخل پوشه `relposenet/` قرار دارند.

## فایل‌هایی که باید جایگزین شوند

این چهار فایل را در پروژه خودت جایگزین کن:

- `relposenet/augmentations.py`
- `relposenet/dataset.py`
- `relposenet/criterion.py`
- `relposenet/pipeline.py`

فایل جدید `relposenet/losses.py` را نیز به پروژه اضافه کن.

`main.py` و `relposenet/model.py` تغییر نمی‌کنند. `SevenScenesTestDataset` و تابع قدیمی `get_augmentations()` برای سازگاری با کد ارزیابی قبلی حفظ شده‌اند.

## چه Lossهایی قابل انتخاب‌اند؟

- `mse`: همان Loss اولیه پروژه، یعنی `MSE(t) + alpha * MSE(q)`؛ بدون تغییر فرمول و وزن‌دهی.
- `hard`: فاصله Focus Pointها.
- `soft`: نسخه Soft با `zf` و `sigma`.
- `softgrid`: نمونه‌برداری Gaussian اطراف چهار گوشه تصویر.
- `dir`: ablation جهت پرتوها.
- `pmse`: Pose-MSE با هم‌ترازی علامت Quaternion.
- `posenet10`, `posenet100`, `posenet`: Baseline نوع PoseNet.
- `se3`: خطای Translation و زاویه ژئودزیک SO(3).
- `ray`, `denseray`: خطای Plücker روی Grid و Dense Grid.
- `soft_sqrt`, `softgrid_sqrt`: نسخه‌های ریشه‌دار.

`Reproj-px` عمداً اضافه نشده، چون در این آزمایش نمی‌خواهی از نقاط سه‌بعدی صحنه استفاده کنی.

## تغییرات هندسی لازم

فایل‌های رسمی 7-Scenes برای هر فریم، `frame-XXXXXX.pose.txt` دارند. این فایل‌ها ماتریس 4x4 دوربین به جهان (`camera-to-world`) را ذخیره می‌کنند. برای اعتبارسنجی برچسب‌های زوج و گزارش خطای انتقال واقعاً متریک، این پچ در **تمام اجراها، حتی baseline MSE**، فایل‌های pose مطلق را می‌خواند و قرارداد زوج‌ها را روی حداکثر ۹۶ زوج از هر split بررسی می‌کند. برای Lossهای هندسی از Rotation دوربین اول استفاده می‌کنیم تا Translation نسبی را از مختصات جهان به مختصات دوربین اول ببریم:

`R_rel = R1.T @ R2`

`t_rel = R1.T @ (C2 - C1)`

شبکه خروجی همان `q_est, t_est` نسبی را می‌دهد. پیش از آموزش هندسی، Dataset قرارداد Quaternion و علامت Translation در فایل pair را با Poseهای مطلق چند زوج بررسی می‌کند. اگر قرارداد فایل‌ها با این فرض‌ها سازگار نباشد، آموزش متوقف می‌شود و به‌جای محاسبه Loss اشتباه، خطای توضیحی می‌دهد.

بنابراین **همه Lossها** به فایل‌های `frame-XXXXXX.pose.txt` نیاز دارند تا معنای متریک برچسب‌ها قبل از مقایسه تأیید شود. اگر این فایل‌ها کنار تصاویر نیستند، `data_params.pose_dir` را در config اضافه کن تا به پوشه‌ای اشاره کند که همان ساختار `scene/seq/frame-XXXXXX.pose.txt` را دارد. اگر فایل زوج‌ها واقعاً از این قرارداد استفاده نکند (مثلاً Translation واحد یا در چارچوب دیگری باشد)، برنامه عمداً متوقف می‌شود؛ در آن صورت باید قرارداد واقعی فایل زوج را بررسی کنیم، نه اینکه یک تبدیل حدسی اعمال کنیم.

## Intrinsics و Augmentation

در کد قبلی، `RandomCrop` برای دو تصویر زوج به‌طور مستقل اجرا می‌شد. این با فرض یک `K` مشترک برای دو دوربین سازگار نیست، چون مختصات principal point دو crop متفاوت می‌شد. نسخه جدید یک Crop مشترک برای هر زوج انتخاب می‌کند و `cx,cy` را مطابق محل Crop به‌روزرسانی می‌کند. در نتیجه `K` برای هر دو تصویر یک زوج مشترک است، اما با تغییر محل Crop بین زوج‌ها مقدار `K` crop-adjusted می‌تواند تغییر کند.

**محدودیت کالیبراسیون:** طبق توضیح رسمی Microsoft، تصاویر RGB و Depth در 7-Scenes کالیبره‌شده نسبت به هم نیستند و پارامتر رسمی کالیبراسیون RGB ارائه نشده است. مقادیر پیش‌فرض این پچ (`fx=532.57, fy=531.54, cx=320, cy=240`) از یک برآورد مستقل RGB مبتنی بر نقاط گریز گرفته شده‌اند؛ این برآورد نیز کالیبراسیون رسمی یا تضمین‌شده دیتاست نیست. مقادیر را در `camera_params` قابل‌تغییر گذاشته‌ایم. برای نتیجه‌گیری علمی قوی درباره Lossهای مبتنی بر پرتو، باید K واقعی RGB را کالیبره کنی یا نتایج را صریحاً تحت فرض K تقریبی گزارش کنی.

تصاویر باید مطابق دستورالعمل RelPoseNet پیش‌پردازش شده باشند: ضلع کوچک‌تر 256 پیکسل و نسبت ابعاد حفظ‌شده. سپس Crop 224 انجام می‌شود. کد این اندازه را بررسی می‌کند و اگر تصاویر خام یا با ابعاد دیگر باشند، خطا می‌دهد.

## تغییر config

کلیدهای `camera_params` را از `configs/main_config_additions.yaml` به ریشه `configs/main.yaml` اضافه کن. کلیدهای `loss_name`, `loss_zf`, `loss_sigma`, `loss_rho`, `loss_beta`, `run_id` را به بخش موجود `train_params` اضافه کن؛ بخش فعلی را حذف یا جایگزین نکن.

برای مثال:

```yaml
camera_params:
  fx: 532.57
  fy: 531.54
  cx: 320.0
  cy: 240.0
  raw_width: 640
  raw_height: 480
  output_size: 224
  expected_short_side: 256

train_params:
  # بقیه تنظیمات قبلی را نگه دار
  loss_name: mse
  loss_zf: 1.0
  loss_sigma: 4.0
  loss_rho: 24.0
  loss_beta: 100.0
  run_id: ""
```

اگر poseها کنار تصاویر resized نیستند:

```yaml
data_params:
  pose_dir: /content/7scenes_original
```

`pose_dir` باید همان پوشه‌بندی صحنه‌ها و sequenceها را داشته باشد. هر فایل pose باید ماتریس Camera-to-World همان فریم RGB باشد.

## اجرای جداگانه هر Loss در Colab

هر دستور یک فرآیند مستقل می‌سازد و از seed و مقداردهی اولیه یکسان استفاده می‌کند. برای مقایسه منصفانه، هر اجرا باید از ابتدا train شود؛ `model_params.resume_snapshot` را `null` نگه دار.

```bash
python main.py train_params.loss_name=mse
python main.py train_params.loss_name=hard train_params.loss_zf=1.0
python main.py train_params.loss_name=soft train_params.loss_zf=1.0 train_params.loss_sigma=4.0
python main.py train_params.loss_name=softgrid train_params.loss_zf=1.0 train_params.loss_sigma=4.0 train_params.loss_rho=24.0
python main.py train_params.loss_name=dir
python main.py train_params.loss_name=pmse
python main.py train_params.loss_name=posenet10
python main.py train_params.loss_name=posenet100
python main.py train_params.loss_name=se3
python main.py train_params.loss_name=ray
python main.py train_params.loss_name=denseray
python main.py train_params.loss_name=soft_sqrt train_params.loss_zf=1.0 train_params.loss_sigma=4.0
python main.py train_params.loss_name=softgrid_sqrt train_params.loss_zf=1.0 train_params.loss_sigma=4.0 train_params.loss_rho=48.0
```

در Colab می‌توانی همین دستورها را با `!` اجرا کنی. برای هر اجرا، مقادیر `n_train_iters`, `bs`, `lr`, scheduler و فواصل validation از config فعلی خوانده می‌شوند؛ تعداد iterationها به epoch تبدیل نمی‌شود و تغییر نمی‌کند.

## خروجی‌ها

هر Loss پوشه جداگانه‌ای در مسیر زیر می‌سازد:

`output_params.out_dir/loss_comparison/<loss-and-parameters>/`

داخل آن:

- `tensorboard/`: لاگ Loss آموزش/اعتبارسنجی و معیارهای Pose.
- `checkpoints/latest.pth`: آخرین checkpoint در زمان validation.
- `checkpoints/final.pth`: checkpoint نهایی.
- `metrics.csv`: معیارهای validation در هر مرحله.
- `loss_curves.png`: نمودار Train/Validation Loss همان تابع هدف.
- `translation_accuracy.png`: میانه خطای انتقال برحسب متر.
- `rotation_accuracy.png`: میانه خطای چرخش برحسب درجه.

برای مشاهده TensorBoard:

```python
%load_ext tensorboard
%tensorboard --logdir /content/your_output_dir/loss_comparison
```


## مقایسه نهایی چند اجرای Loss

بعد از اجرای چند Loss، این اسکریپت را اجرا کن تا CSV نهایی و نمودارهای مقایسه‌ای روی معیارهای مشترک ساخته شوند:

```bash
python compare_loss_runs.py --root outputs/loss_comparison
```

در Colab نیز مسیر خروجی را بده، مثلاً:

```python
!python compare_loss_runs.py --root /content/outputs/loss_comparison
```

خروجی‌ها داخل پوشه `comparison/` هستند: `final_comparison.csv`، نمودارهای مقایسه Translation/Rotation نهایی و منحنی‌های دقت Validation در طول آموزش. مقدار خام Loss بین روش‌های مختلف روی یک محور رسم نمی‌شود، چون مقیاس و واحد تابع‌های هدف متفاوت است.

## معیار مقایسه

Lossهای مختلف مقدار عددی و مقیاس متفاوت دارند؛ بنابراین مقدار `val_loss` بین دو Loss متفاوت معیار دقت نیست. معیار اصلی مقایسه:

- Median translation error (m)
- Median rotation error (degrees)
- Median translation-direction error (degrees)

همه این معیارها روی همان زوج‌های Validation و با همان پیش‌بینی‌های شبکه محاسبه می‌شوند. `mse` دقیقاً Baseline قبلی است. در این پچ checkpointها از روی مقدار Loss بین روش‌ها مقایسه نمی‌شوند؛ هر اجرا تا همان `n_train_iters` ادامه پیدا می‌کند و checkpoint نهایی ذخیره می‌شود.

## منابع دیتاست و هندسه

- صفحه رسمی Microsoft 7-Scenes و توضیح فرمت pose و محدودیت RGB/Depth calibration: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
- برآورد مستقل intrinsics دوربین RGB با نقاط گریز (منبع مقادیر پیش‌فرض `532.57, 531.54, 320, 240`): https://github.com/zinsmatt/7-Scenes-Calibration
- مخزن رسمی PyTorch RelPoseNet و دستور resize تصاویر: https://github.com/AaltoVision/RelPoseNet
- مقاله RelPoseNet: https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Laskar_Camera_Relocalization_by_Computing_Pairwise_Relative_Poses_Using_Convolutional_Neural_Network_ICCV_2017_paper.pdf

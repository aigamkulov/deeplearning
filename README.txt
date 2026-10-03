# Deep Learning Practice — MODERN CMS

Bu loyiha Flask asosida yaratilgan Deep Learning amaliy mashg‘ulotlar platformasidir.

## Asosiy imkoniyatlar

- 1-mavzu va 2-mavzu saqlangan.
- Mavzular `data/exercises.json` orqali dinamik boshqariladi.
- Web sahifaning o‘zidan yangi mavzu qo‘shish mumkin.
- Web sahifaning o‘zidan yangi masala qo‘shish mumkin.
- Yangi masala uchun avtomatik raqam beriladi: 31, 32, 33...
- Python kodi avtomatik `exercises/31.py` kabi faylga yoziladi.
- Matematik formulalar MathJax bilan chiroyli ko‘rsatiladi.
- Formula -> Python kodi bog‘lanishini kiritish mumkin.
- Masala sharti va yechish algoritmi saqlanadi.
- RUN tugmasi orqali Python kodi bajariladi.
- Dark/Light rejim mavjud.
- Responsive zamonaviy dizayn.

## Ishga tushirish

Windows CMD:

    cd "Deep Learning"
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python app.py

Agar sizda loyiha uchun eski `venv` allaqachon mavjud bo‘lsa:

    venv\Scripts\activate
    pip install -r requirements.txt
    python app.py

Brauzer:

    http://127.0.0.1:5000

## Yangi mavzu qo‘shish

Brauzerda:

    http://127.0.0.1:5000/admin

"Yangi mavzu qo‘shish" formasini to‘ldiring.

## Yangi masala qo‘shish

Admin sahifasida:

1. Mavzuni tanlang.
2. Masala nomini yozing.
3. Tavsif va masala shartini kiriting.
4. Matematik formulalarni har birini yangi qatorda yozing.
5. Algoritm qadamlarini har birini yangi qatorda yozing.
6. Formula -> Python kod qatorlarini kiriting.
7. Python kodini yozing.
8. "Masalani saqlash" tugmasini bosing.

Masala avtomatik raqamlanadi va tanlangan mavzuda paydo bo‘ladi.

## Muhim xavfsizlik eslatmasi

`/run/<number>` serverda Python faylini bajaradi. Shu sababli bu versiya lokal o‘quv muhiti uchun mo‘ljallangan.

Agar sayt internetga chiqarilsa, admin login/parol, CSRF himoyasi, fayl tekshiruvi va Python kodini sandbox/container ichida ishga tushirish mexanizmini qo‘shish kerak.

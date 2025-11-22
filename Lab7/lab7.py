import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QSlider, QComboBox, QGroupBox,
                             QMessageBox, QLineEdit, QRadioButton, QButtonGroup, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer
import cv2
import time

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.original_array = None
        self.current_array = None
        self.current_scale = 1.0
        
    def initUI(self):
        self.setWindowTitle('Лабораторная работа 7 - Масштабирование изображений')
        self.setGeometry(100, 100, 1400, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Панель с кнопками управления
        btn_layout = QHBoxLayout()
        
        load_btn = QPushButton('Загрузить изображение')
        load_btn.clicked.connect(self.load_image)
        btn_layout.addWidget(load_btn)
        
        reset_btn = QPushButton('Сбросить')
        reset_btn.clicked.connect(self.reset_image)
        btn_layout.addWidget(reset_btn)
        
        geometric_btn = QPushButton('Геометрические фигуры')
        geometric_btn.clicked.connect(self.create_geometric_image)
        btn_layout.addWidget(geometric_btn)
        
        main_layout.addLayout(btn_layout)
        
        # Панель для отображения изображений (оригинал и результат)
        images_layout = QHBoxLayout()
        
        # Оригинальное изображение с прокруткой
        orig_group = QGroupBox("Оригинальное изображение")
        orig_layout = QVBoxLayout()
        
        self.orig_scroll = QScrollArea()
        self.orig_label = QLabel('Оригинал')
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setMinimumSize(600, 600)
        self.orig_label.setStyleSheet("border: 1px solid gray;")
        self.orig_scroll.setWidget(self.orig_label)
        self.orig_scroll.setAlignment(Qt.AlignCenter)
        orig_layout.addWidget(self.orig_scroll)
        orig_group.setLayout(orig_layout)
        images_layout.addWidget(orig_group)
        
        # Результат масштабирования с прокруткой
        result_group = QGroupBox("Результат масштабирования")
        result_layout = QVBoxLayout()
        
        self.result_scroll = QScrollArea()
        self.result_label = QLabel('Результат')
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(600, 600)
        self.result_label.setStyleSheet("border: 1px solid gray;")
        self.result_scroll.setWidget(self.result_label)
        self.result_scroll.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.result_scroll)
        
        result_group.setLayout(result_layout)
        images_layout.addWidget(result_group)
        
        main_layout.addLayout(images_layout)
        
        # Панель управления
        control_layout = QHBoxLayout()
        
        # Метод интерполяции
        method_group = QGroupBox("Метод интерполяции")
        method_layout = QVBoxLayout()
        
        self.method_group = QButtonGroup(self)
        self.nearest_rb = QRadioButton("Ближайший сосед")
        self.bicubic_rb = QRadioButton("Бикубическая")
        
        self.nearest_rb.setChecked(True)
        
        self.method_group.addButton(self.nearest_rb)
        self.method_group.addButton(self.bicubic_rb)
        
        method_layout.addWidget(self.nearest_rb)
        method_layout.addWidget(self.bicubic_rb)
        method_group.setLayout(method_layout)
        control_layout.addWidget(method_group)
        
        # Коэффициенты масштабирования
        scale_group = QGroupBox("Масштабирование")
        scale_layout = QVBoxLayout()
        
        # Быстрые кнопки для демонстрации
        quick_buttons_layout = QHBoxLayout()
        
        scale_025_btn = QPushButton('×0.25')
        scale_025_btn.clicked.connect(lambda: self.apply_quick_scale(0.25))
        quick_buttons_layout.addWidget(scale_025_btn)
        
        scale_05_btn = QPushButton('×0.5')
        scale_05_btn.clicked.connect(lambda: self.apply_quick_scale(0.5))
        quick_buttons_layout.addWidget(scale_05_btn)
        
        scale_2_btn = QPushButton('×2')
        scale_2_btn.clicked.connect(lambda: self.apply_quick_scale(2.0))
        quick_buttons_layout.addWidget(scale_2_btn)
        
        scale_4_btn = QPushButton('×4')
        scale_4_btn.clicked.connect(lambda: self.apply_quick_scale(4.0))
        quick_buttons_layout.addWidget(scale_4_btn)
        
        scale_layout.addLayout(quick_buttons_layout)
        
        # Ручной ввод
        manual_layout = QHBoxLayout()
        manual_layout.addWidget(QLabel('Коэффициент:'))
        self.scale_input = QLineEdit()
        self.scale_input.setText('1.0')
        self.scale_input.setMaximumWidth(60)
        manual_layout.addWidget(self.scale_input)
        
        apply_btn = QPushButton('Применить')
        apply_btn.clicked.connect(self.apply_scaling)
        manual_layout.addWidget(apply_btn)
        
        scale_layout.addLayout(manual_layout)
        
        # Информация
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel('Масштаб:'))
        self.scale_label = QLabel('1.0x')
        info_layout.addWidget(self.scale_label)
        info_layout.addStretch()
        
        scale_layout.addLayout(info_layout)
        scale_group.setLayout(scale_layout)
        control_layout.addWidget(scale_group)
        
        main_layout.addLayout(control_layout)
    
    def apply_quick_scale(self, scale_factor):
        """Быстрое применение масштабирования с предустановленным коэффициентом"""
        self.scale_input.setText(str(scale_factor))
        self.apply_scaling()
    
    def apply_scaling(self):
        """Применение масштабирования"""
        if self.original_array is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение")
            return
            
        try:
            scale_factor = float(self.scale_input.text())
            if scale_factor <= 0:
                raise ValueError("Коэффициент должен быть положительным")
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", f"Некорректный коэффициент: {e}")
            return
        
        if self.nearest_rb.isChecked():
            scaled_image = self.optimized_nearest_neighbor(self.original_array, scale_factor)
            method_name = "ближайшего соседа"
        else:
            scaled_image = self.bicubic_interpolation_wikipedia(self.original_array, scale_factor)
            method_name = "бикубической интерполяции"
        
        self.current_array = scaled_image
        self.current_scale = scale_factor
        self.scale_label.setText(f"{scale_factor:.2f}x ({method_name})")
        
        self.update_display()
    
    def optimized_nearest_neighbor(self, image, scale_factor):
        """Оптимизированный метод ближайшего соседа"""
        h, w = image.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        # Используем векторные операции NumPy для ускорения
        x_indices = (np.arange(new_w) / scale_factor).astype(int)
        y_indices = (np.arange(new_h) / scale_factor).astype(int)
        
        # Ограничиваем индексы
        x_indices = np.clip(x_indices, 0, w-1)
        y_indices = np.clip(y_indices, 0, h-1)
        
        # Создаем сетку индексов и применяем индексирование
        scaled = image[np.ix_(y_indices, x_indices)]
        
        return scaled
    
    def bicubic_interpolation_wikipedia(self, image, scale_factor):
        """Бикубическая интерполяция по формулам из Википедии"""
        h, w = image.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        # Создаем новое изображение
        scaled = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        
        # Преобразуем в float для точных вычислений
        image_float = image.astype(np.float32)
        
        for i in range(new_h):
            for j in range(new_w):
                # Вычисляем координаты в исходном изображении
                x = j / scale_factor
                y = i / scale_factor
                
                # Находим базовые координаты
                x0 = int(x)
                y0 = int(y)
                
                # Вычисляем дробные части
                dx = x - x0
                dy = y - y0
                
                # Получаем значения 16 соседних пикселей
                pixels = self.get_surrounding_pixels(image_float, x0, y0, w, h)
                
                # Вычисляем коэффициенты по формулам из Википедии
                # и применяем их к соответствующим пикселям
                pixel_value = (
                    0.25 * (dx-1)*(dx-2)*(dx+1) * (dy-1)*(dy-2)*(dy+1) * pixels[1, 1] +  
                    -0.25 * dx*(dx+1)*(dx-2) * (dy-1)*(dy-2)*(dy+1) * pixels[1, 2] +     
                    -0.25 * dy*(dx-1)*(dx-2)*(dx+1) * (dy+1)*(dy-2) * pixels[2, 1] +    
                    0.25 * dx*dy*(dx+1)*(dx-2) * (dy+1)*(dy-2) * pixels[2, 2] +         
                    -1/12 * dx*(dx-1)*(dx-2) * (dy-1)*(dy-2)*(dy+1) * pixels[1, 0] +     
                    -1/12 * dy*(dx-1)*(dx-2)*(dx+1) * (dy-1)*(dy-2) * pixels[0, 1] +     
                    1/12 * dx*dy*(dx-1)*(dx-2) * (dy+1)*(dy-2) * pixels[2, 0] +         
                    1/12 * dx*dy*(dx+1)*(dx-2) * (dy-1)*(dy-2) * pixels[0, 2] +         
                    1/12 * dx*(dx-1)*(dx+1) * (dy-1)*(dy-2)*(dy+1) * pixels[1, 3] +     
                    1/12 * dy*(dx-1)*(dx-2)*(dx+1) * (dy-1)*(dy+1) * pixels[3, 1] +      
                    1/36 * dx*dy*(dx-1)*(dx-2) * (dy-1)*(dy-2) * pixels[0, 0] +          
                    -1/12 * dx*dy*(dx-1)*(dx+1) * (dy+1)*(dy-2) * pixels[2, 3] +         
                    -1/12 * dx*dy*(dx+1)*(dx-2) * (dy-1)*(dy+1) * pixels[3, 2] +         
                    -1/36 * dx*dy*(dx-1)*(dx+1) * (dy-1)*(dy-2) * pixels[0, 3] +        
                    -1/36 * dx*dy*(dx-1)*(dx-2) * (dy+1)*(dy-1) * pixels[3, 0] +         
                    1/36 * dx*dy*(dx-1)*(dx+1) * (dy-1)*(dy+1) * pixels[3, 3]            
                )
                
                # Обрезаем значения и преобразуем обратно в uint8
                pixel_value = np.clip(pixel_value, 0, 255)
                scaled[i, j] = pixel_value.astype(np.uint8)
        
        return scaled
    
    def get_surrounding_pixels(self, image, x, y, w, h):
        """Получаем 16 окружающих пикселей для бикубической интерполяции"""
        pixels = np.zeros((4, 4, 3), dtype=np.float32)
        
        for i in range(-1, 3):
            for j in range(-1, 3):
                # Координаты в исходном изображении
                src_x = x + j
                src_y = y + i
                
                # Обработка границ - используем отражение
                src_x = self.reflect_border(src_x, w)
                src_y = self.reflect_border(src_y, h)
                
                pixels[i+1, j+1] = image[src_y, src_x]
        
        return pixels
    
    def reflect_border(self, coord, size):
        """Отражает координаты за границами изображения"""
        if coord < 0:
            return -coord - 1
        elif coord >= size:
            return 2 * size - coord - 1
        else:
            return coord
    
    def create_geometric_image(self):
        """Создание тестового изображения с геометрическими фигурами"""
        # Создаем изображение меньшего размера для быстрой демонстрации
        width, height = 200, 200
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Рисуем круги
        cv2.circle(image, (50, 50), 25, (255, 0, 0), -1)
        cv2.circle(image, (150, 50), 25, (0, 255, 0), 2)
        
        # Рисуем прямоугольники
        cv2.rectangle(image, (25, 100), (75, 150), (0, 0, 255), -1)
        cv2.rectangle(image, (125, 100), (175, 150), (255, 255, 0), 2)
        
        # Рисуем линии
        cv2.line(image, (25, 175), (175, 175), (0, 0, 0), 2)
        cv2.line(image, (100, 25), (100, 175), (0, 0, 0), 2)
        
        # Рисуем кольца
        for r in range(10, 40, 8):
            cv2.circle(image, (150, 150), r, (128, 0, 128), 1)
        
        self.original_array = image
        self.current_array = image.copy()
        self.current_scale = 1.0
        self.scale_input.setText("1.0")
        self.scale_label.setText("1.0x")
        self.update_display()
    
    def load_image(self):
        """Загрузка изображения из файла"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", 
            "Image Files (*.png *.jpg *.jpeg)")
        
        if file_name:
            # Загружаем изображение
            image = cv2.imread(file_name)
            if image is None:
                QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение")
                return
            
            # Конвертируем в RGB
            self.original_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_array = self.original_array.copy()
            self.current_scale = 1.0
            self.scale_input.setText("1.0")
            self.scale_label.setText("1.0x")
            
            self.update_display()
    
    def update_display(self):
        """Обновление отображения изображений"""
        if self.original_array is not None:
            # Отображаем оригинал
            orig_qimage = self.array_to_qimage(self.original_array)
            orig_pixmap = QPixmap.fromImage(orig_qimage)
            self.orig_label.setPixmap(orig_pixmap)
            self.orig_label.resize(orig_pixmap.size())
            
            # Отображаем результат
            if self.current_array is not None:
                result_qimage = self.array_to_qimage(self.current_array)
                result_pixmap = QPixmap.fromImage(result_qimage)
                self.result_label.setPixmap(result_pixmap)
                self.result_label.resize(result_pixmap.size())
    
    def array_to_qimage(self, array):
        """Преобразование numpy array в QImage"""
        height, width, channels = array.shape
        bytes_per_line = channels * width
        return QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    def reset_image(self):
        """Сброс к оригинальному изображению"""
        if self.original_array is not None:
            self.current_array = self.original_array.copy()
            self.current_scale = 1.0
            self.scale_input.setText("1.0")
            self.scale_label.setText("1.0x")
            self.update_display()

app = QApplication(sys.argv)
viewer = ImageViewer()
viewer.show()
sys.exit(app.exec_())
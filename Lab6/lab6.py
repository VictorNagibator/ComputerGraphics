import sys
import numpy as np
import random
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QComboBox, QGroupBox,
                             QMessageBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.original_array = None
        self.current_array = None
        
    def initUI(self):
        self.setWindowTitle('Лабораторная работа 6')
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Левая панель - изображение
        left_panel = QVBoxLayout()
        
        self.image_label = QLabel('Изображение не загружено')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        left_panel.addWidget(self.image_label)
        
        # Кнопки управления
        btn_layout = QHBoxLayout()
        
        load_btn = QPushButton('Загрузить изображение')
        load_btn.clicked.connect(self.load_image)
        btn_layout.addWidget(load_btn)
        
        reset_btn = QPushButton('Сбросить изменения')
        reset_btn.clicked.connect(self.reset_image)
        btn_layout.addWidget(reset_btn)
        
        left_panel.addLayout(btn_layout)
        
        # Правая панель - фильтры
        right_panel = QVBoxLayout()
        
        # Группа для наложения шума
        noise_group = QGroupBox('Искусственное наложение шума')
        noise_layout = QVBoxLayout()
        
        noise_type_layout = QHBoxLayout()
        noise_type_layout.addWidget(QLabel('Тип шума:'))
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(['Точки', 'Линии', 'Окружности'])
        noise_type_layout.addWidget(self.noise_combo)
        
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel('Интенсивность:'))
        self.noise_intensity = QSpinBox()
        self.noise_intensity.setRange(1, 50)
        self.noise_intensity.setValue(10)
        intensity_layout.addWidget(self.noise_intensity)
        
        apply_noise_btn = QPushButton('Наложить шум')
        apply_noise_btn.clicked.connect(self.apply_noise)
        
        noise_layout.addLayout(noise_type_layout)
        noise_layout.addLayout(intensity_layout)
        noise_layout.addWidget(apply_noise_btn)
        noise_group.setLayout(noise_layout)
        right_panel.addWidget(noise_group)
        
        # Группа для фильтров шумоподавления
        denoise_group = QGroupBox('Фильтры шумоподавления')
        denoise_layout = QVBoxLayout()
        
        denoise_type_layout = QHBoxLayout()
        denoise_type_layout.addWidget(QLabel('Тип фильтра:'))
        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems(['Равномерный', 'Медианный'])
        denoise_type_layout.addWidget(self.denoise_combo)
        
        aperture_layout = QHBoxLayout()
        aperture_layout.addWidget(QLabel('Размер апертуры:'))
        self.aperture_size = QSpinBox()
        self.aperture_size.setRange(3, 15)
        self.aperture_size.setValue(3)
        self.aperture_size.setSingleStep(2)
        aperture_layout.addWidget(self.aperture_size)
        
        apply_denoise_btn = QPushButton('Применить фильтр')
        apply_denoise_btn.clicked.connect(self.apply_denoise)
        
        denoise_layout.addLayout(denoise_type_layout)
        denoise_layout.addLayout(aperture_layout)
        denoise_layout.addWidget(apply_denoise_btn)
        denoise_group.setLayout(denoise_layout)
        right_panel.addWidget(denoise_group)
        
        # Группа для повышения резкости
        sharpen_group = QGroupBox('Повышение резкости')
        sharpen_layout = QVBoxLayout()
        
        sharpen_method_layout = QHBoxLayout()
        sharpen_method_layout.addWidget(QLabel('Метод:'))
        self.sharpen_combo = QComboBox()
        self.sharpen_combo.addItems(['Простое ядро', 'Лапласиан', 'Фильтр Собеля', 'Нерезкое маскирование'])
        sharpen_method_layout.addWidget(self.sharpen_combo)
        
        sharpen_strength_layout = QHBoxLayout()
        sharpen_strength_layout.addWidget(QLabel('Степень резкости:'))
        self.sharpen_strength = QSpinBox()
        self.sharpen_strength.setRange(1, 10)
        self.sharpen_strength.setValue(2)
        sharpen_strength_layout.addWidget(self.sharpen_strength)
        
        apply_sharpen_btn = QPushButton('Увеличить резкость')
        apply_sharpen_btn.clicked.connect(self.apply_sharpening)
        
        sharpen_layout.addLayout(sharpen_method_layout)
        sharpen_layout.addLayout(sharpen_strength_layout)
        sharpen_layout.addWidget(apply_sharpen_btn)
        sharpen_group.setLayout(sharpen_layout)
        right_panel.addWidget(sharpen_group)
        
        # Группа для спецэффектов
        effects_group = QGroupBox('Спецэффекты')
        effects_layout = QVBoxLayout()
        
        # Параметры для эффекта волн
        waves_params_layout = QVBoxLayout()
        
        amplitude_layout = QHBoxLayout()
        amplitude_layout.addWidget(QLabel('Амплитуда:'))
        self.waves_amplitude = QSpinBox()
        self.waves_amplitude.setRange(1, 50)
        self.waves_amplitude.setValue(10)
        amplitude_layout.addWidget(self.waves_amplitude)
        
        frequency_layout = QHBoxLayout()
        frequency_layout.addWidget(QLabel('Частота:'))
        self.waves_frequency = QDoubleSpinBox()
        self.waves_frequency.setRange(0.01, 0.2)
        self.waves_frequency.setValue(0.05)
        self.waves_frequency.setSingleStep(0.01)
        frequency_layout.addWidget(self.waves_frequency)
        
        waves_btn = QPushButton('Эффект волны')
        waves_btn.clicked.connect(self.apply_waves)
        
        waves_params_layout.addLayout(amplitude_layout)
        waves_params_layout.addLayout(frequency_layout)
        waves_params_layout.addWidget(waves_btn)
        
        effects_layout.addLayout(waves_params_layout)
        effects_group.setLayout(effects_layout)
        right_panel.addWidget(effects_group)
        
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
    
    def apply_noise(self):
        if self.current_array is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение")
            return
            
        noise_type = self.noise_combo.currentText()
        intensity = self.noise_intensity.value()
        
        noisy_image = self.current_array.copy()
        height, width = noisy_image.shape[:2]
        
        if noise_type == 'Точки':
            # Импульсный шум (точки)
            for _ in range(intensity * 100):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                color = (255,255,255)
                cv2.circle(noisy_image, (x, y), 1, color, -1)
                
        elif noise_type == 'Линии':
            # Линейный шум
            for _ in range(intensity * 5):
                x1 = random.randint(0, width - 1)
                y1 = random.randint(0, height - 1)
                x2 = random.randint(0, width - 1)
                y2 = random.randint(0, height - 1)
                color = (255,255,255)
                thickness = 1
                cv2.line(noisy_image, (x1, y1), (x2, y2), color, thickness)
                
        elif noise_type == 'Окружности':
            # Шум в виде окружностей
            for _ in range(intensity * 10):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                radius = random.randint(3, 7)
                color = (255,255,255)
                thickness = 1
                cv2.circle(noisy_image, (x, y), radius, color, thickness)
        
        self.current_array = noisy_image
        self.update_display()
    
    def apply_denoise(self):
        if self.current_array is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение")
            return
            
        filter_type = self.denoise_combo.currentText()
        aperture = self.aperture_size.value()
        
        # Убедимся, что размер апертуры нечетный
        if aperture % 2 == 0:
            aperture += 1
            self.aperture_size.setValue(aperture)
        
        if filter_type == 'Равномерный':
            # Равномерный фильтр (усредняющий)
            kernel = np.ones((aperture, aperture), np.float32) / (aperture * aperture)
            filtered = cv2.filter2D(self.current_array, -1, kernel)
            
        elif filter_type == 'Медианный':
            # Медианный фильтр
            filtered = cv2.medianBlur(self.current_array, aperture)
        
        self.current_array = filtered
        self.update_display()
    
    def apply_sharpening(self):
        if self.current_array is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение")
            return
            
        method = self.sharpen_combo.currentText()
        k = self.sharpen_strength.value()
        
        if method == 'Простое ядро':
            # Ядро для повышения резкости согласно лекции
            kernel = np.full((3,3), -k/8.0, dtype=np.float32)
            kernel[1,1] = 1.0 + k
            sharpened = cv2.filter2D(self.current_array, -1, kernel, borderType=cv2.BORDER_DEFAULT)
            self.current_array = sharpened
            
        elif method == 'Лапласиан':
            # Лапласиан для выделения границ
            kernel = np.array([[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]], dtype=np.float32)
            
            # Применяем лапласиан и добавляем к исходному изображению
            edges = cv2.filter2D(self.current_array, -1, kernel)
            self.current_array = cv2.addWeighted(self.current_array, 1, edges, k/10, 0)
            
        elif method == 'Фильтр Собеля':
            # Фильтр Собеля для выделения границ
            sobel_x = cv2.Sobel(self.current_array, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(self.current_array, cv2.CV_64F, 0, 1, ksize=3)
            
            # Объединяем градиенты
            sobel = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel = np.uint8(sobel / np.max(sobel) * 255)
            
            # Добавляем границы к исходному изображению
            self.current_array = cv2.addWeighted(self.current_array, 1, sobel, k/20, 0)
            
        elif method == 'Нерезкое маскирование':
            # размываем изображение и вычитаем из оригинала
            blurred = cv2.GaussianBlur(self.current_array, (0, 0), 3)
            sharpened = cv2.addWeighted(self.current_array, 1 + k/10, blurred, -k/10, 0)
            self.current_array = sharpened
        
        self.update_display()
    
    def apply_waves(self):
        if self.current_array is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение")
            return
            
        height, width = self.current_array.shape[:2]
        result = np.zeros_like(self.current_array)
        
        # Получаем параметры из интерфейса
        amplitude = self.waves_amplitude.value()
        frequency = self.waves_frequency.value()
        
        for y in range(height):
            for x in range(width):
                # Создаем волнообразное искажение
                dx = int(amplitude * math.sin(2 * math.pi * frequency * y))
                new_x = (x + dx) % width
                result[y, x] = self.current_array[y, new_x]
        
        self.current_array = result
        self.update_display()

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", 
            "Image Files (*.png *.jpg *.jpeg)")
        
        if file_name:
            self.original_array = cv2.imread(file_name)
            if self.original_array is None:
                QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение")
                return
            
            self.original_array = cv2.cvtColor(self.original_array, cv2.COLOR_BGR2RGB)
            self.current_array = self.original_array.copy()
            self.update_display()
    
    def reset_image(self):
        if self.original_array is not None:
            self.current_array = self.original_array.copy()
            self.update_display()
    
    def array_to_qimage(self, array):
        height, width, channels = array.shape
        bytes_per_line = channels * width
        return QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    def update_display(self):
        if self.current_array is not None:
            # Масштабируем изображение для отображения
            h, w = self.current_array.shape[:2]
            max_display_size = 800
            if max(h, w) > max_display_size:
                scale = max_display_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                display_array = cv2.resize(self.current_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                display_array = self.current_array
            
            # Преобразуем в QImage и QPixmap
            qimage = self.array_to_qimage(display_array)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap)


app = QApplication(sys.argv)
viewer = ImageViewer()
viewer.show()
sys.exit(app.exec_())
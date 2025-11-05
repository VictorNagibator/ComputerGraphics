import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QSlider, QComboBox, QGroupBox,
                             QMessageBox, QCheckBox)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QRect, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import cv2

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.original_image = None
        self.current_image = None
        self.original_array = None
        self.current_array = None
        self.selection_rect = None
        self.drawing_selection = False
        self.selection_start = None
        self.selection_active = False
        
        # Параметры преобразований
        self.brightness_value = 0
        self.contrast_value = 100
        
        # Для гистограммы
        self.hist_initialized = False
        self.hist_bars = None
        
        # Таймер для отложенного обновления
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.apply_all_transforms)
        
    def initUI(self):
        self.setWindowTitle('Обработка растровых изображений')
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
        self.image_label.mousePressEvent = self.image_mouse_press
        self.image_label.mouseMoveEvent = self.image_mouse_move
        self.image_label.mouseReleaseEvent = self.image_mouse_release
        left_panel.addWidget(self.image_label)
        
        # Кнопки управления
        btn_layout = QHBoxLayout()
        
        load_btn = QPushButton('Загрузить изображение')
        load_btn.clicked.connect(self.load_image)
        btn_layout.addWidget(load_btn)
        
        reset_btn = QPushButton('Сбросить изменения')
        reset_btn.clicked.connect(self.reset_image)
        btn_layout.addWidget(reset_btn)
        
        select_btn = QPushButton('Выделить область')
        select_btn.clicked.connect(self.toggle_selection)
        btn_layout.addWidget(select_btn)
        
        clear_select_btn = QPushButton('Сбросить выделение')
        clear_select_btn.clicked.connect(self.clear_selection)
        btn_layout.addWidget(clear_select_btn)
        
        left_panel.addLayout(btn_layout)
        
        # Правая панель - управление
        right_panel = QVBoxLayout()
        
        # Группа гистограммы
        hist_group = QGroupBox('Гистограмма яркости')
        hist_layout = QVBoxLayout()
        self.hist_canvas = MplCanvas(self, width=6, height=4, dpi=100)
        hist_layout.addWidget(self.hist_canvas)
        hist_group.setLayout(hist_layout)
        right_panel.addWidget(hist_group)
        
        # Группа преобразований яркости/контрастности
        bc_group = QGroupBox('Яркость и контрастность')
        bc_layout = QVBoxLayout()
        
        # Яркость
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel('Яркость:'))
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.sliderPressed.connect(self.slider_pressed)
        self.brightness_slider.sliderReleased.connect(self.slider_released)
        brightness_layout.addWidget(self.brightness_slider)
        self.brightness_value_label = QLabel('0')
        brightness_layout.addWidget(self.brightness_value_label)
        bc_layout.addLayout(brightness_layout)
        
        # Контрастность
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel('Контрастность:'))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(50, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.sliderPressed.connect(self.slider_pressed)
        self.contrast_slider.sliderReleased.connect(self.slider_released)
        contrast_layout.addWidget(self.contrast_slider)
        self.contrast_value_label = QLabel('100')
        contrast_layout.addWidget(self.contrast_value_label)
        bc_layout.addLayout(contrast_layout)
        
        bc_group.setLayout(bc_layout)
        right_panel.addWidget(bc_group)
        
        # Группа изменения цветности
        color_group = QGroupBox('Изменение цветности')
        color_layout = QVBoxLayout()
        
        # Бинаризация
        binary_layout = QVBoxLayout()
        binary_method_layout = QHBoxLayout()
        binary_method_layout.addWidget(QLabel('Метод бинаризации:'))
        self.binary_combo = QComboBox()
        self.binary_combo.addItems(['Фиксированный порог', 'Адаптивный порог', 'Метод Оцу'])
        self.binary_combo.currentTextChanged.connect(self.on_binary_method_changed)
        binary_method_layout.addWidget(self.binary_combo)
        
        self.threshold_layout = QHBoxLayout()
        self.threshold_label = QLabel('Порог:')
        self.threshold_layout.addWidget(self.threshold_label)
        self.binary_threshold_slider = QSlider(Qt.Horizontal)
        self.binary_threshold_slider.setRange(0, 255)
        self.binary_threshold_slider.setValue(128)
        self.binary_threshold_slider.valueChanged.connect(self.binary_threshold_changed)
        self.threshold_layout.addWidget(self.binary_threshold_slider)
        self.binary_threshold_label = QLabel('128')
        self.threshold_layout.addWidget(self.binary_threshold_label)
        
        binary_layout.addLayout(binary_method_layout)
        binary_layout.addLayout(self.threshold_layout)
        
        apply_binary_btn = QPushButton('Применить бинаризацию')
        apply_binary_btn.clicked.connect(self.apply_binarization)
        binary_layout.addWidget(apply_binary_btn)
        
        color_layout.addLayout(binary_layout)
        
        # Кнопки преобразований
        transform_layout = QHBoxLayout()
        
        gray_btn = QPushButton('Оттенки серого')
        gray_btn.clicked.connect(self.apply_grayscale)
        transform_layout.addWidget(gray_btn)
        
        negative_btn = QPushButton('Негатив')
        negative_btn.clicked.connect(self.apply_negative)
        transform_layout.addWidget(negative_btn)
        
        color_layout.addLayout(transform_layout)
        color_group.setLayout(color_layout)
        right_panel.addWidget(color_group)
        
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
        # Изначально скрываем ползунок порога для нефиксированных методов
        self.on_binary_method_changed(self.binary_combo.currentText())
    
    def on_binary_method_changed(self, method):
        """Обработчик изменения метода бинаризации"""
        if method == 'Фиксированный порог':
            self.threshold_label.setVisible(True)
            self.binary_threshold_slider.setVisible(True)
            self.binary_threshold_label.setVisible(True)
        else:
            self.threshold_label.setVisible(False)
            self.binary_threshold_slider.setVisible(False)
            self.binary_threshold_label.setVisible(False)
    
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)")
        
        if file_name:
            # Загружаем через OpenCV для быстрой обработки
            self.original_array = cv2.imread(file_name)
            if self.original_array is None:
                QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение")
                return
            
            # Конвертируем из BGR в RGB
            self.original_array = cv2.cvtColor(self.original_array, cv2.COLOR_BGR2RGB)
            self.current_array = self.original_array.copy()
            
            self.update_display()
            self.reset_sliders()
            self.clear_selection()
            self.init_histogram()
    
    def init_histogram(self):
        """Инициализирует гистограмму при загрузке изображения"""
        if self.current_array is None:
            return
            
        # Вычисляем яркость по формуле из лекции
        r, g, b = self.current_array[:,:,0], self.current_array[:,:,1], self.current_array[:,:,2]
        brightness = 0.299 * r + 0.5876 * g + 0.114 * b
        brightness = brightness.flatten()
        
        # Строим гистограмму
        self.hist_canvas.axes.clear()
        counts, bins, patches = self.hist_canvas.axes.hist(brightness, bins=256, range=(0, 255), alpha=0.7, color='blue')
        self.hist_canvas.axes.set_xlabel('Яркость')
        self.hist_canvas.axes.set_ylabel('Частота')
        self.hist_canvas.axes.set_title('Гистограмма яркости')
        
        # Сохраняем ссылки на столбцы гистограммы для последующего обновления
        self.hist_bars = patches
        self.hist_initialized = True
        
        # Увеличиваем пространство вокруг графика
        self.hist_canvas.fig.tight_layout(pad=2.0)
        self.hist_canvas.draw()
    
    def update_histogram(self):
        """Обновляет данные гистограммы без изменения масштаба и осей"""
        if self.current_array is None or not self.hist_initialized:
            return
            
        # Вычисляем яркость по формуле из лекции
        r, g, b = self.current_array[:,:,0], self.current_array[:,:,1], self.current_array[:,:,2]
        brightness = 0.299 * r + 0.5876 * g + 0.114 * b
        brightness = brightness.flatten()
        
        # Вычисляем новую гистограмму
        counts, bins = np.histogram(brightness, bins=256, range=(0, 255))
        
        # Обновляем высоту столбцов гистограммы
        for i, patch in enumerate(self.hist_bars):
            patch.set_height(counts[i])
        
        # Перерисовываем гистограмму
        self.hist_canvas.draw()
    
    def array_to_qimage(self, array):
        """Быстрое преобразование numpy array в QImage"""
        height, width, channels = array.shape
        bytes_per_line = channels * width
        return QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    def update_display(self):
        """Обновляет отображение изображения и гистограммы"""
        if self.current_array is not None:
            # Создаем копию для отображения (чтобы не портить оригинал)
            display_array = self.current_array.copy()
            
            # Масштабируем изображение для отображения
            h, w = display_array.shape[:2]
            max_display_size = 800
            if max(h, w) > max_display_size:
                scale = max_display_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                display_array = cv2.resize(display_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            qimage = self.array_to_qimage(display_array)
            pixmap = QPixmap.fromImage(qimage)
            
            # Если есть выделение, рисуем его поверх изображения
            if self.selection_active and self.selection_rect:
                # Создаем painter для рисования поверх pixmap
                painter = QPainter(pixmap)
                
                # Настраиваем перо для тонкой пунктирной линии
                pen = QPen(Qt.red)
                pen.setWidth(1)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                
                # Масштабируем координаты выделения к отображаемому размеру
                scale_x = pixmap.width() / self.current_array.shape[1]
                scale_y = pixmap.height() / self.current_array.shape[0]
                
                x = int(self.selection_rect[0] * scale_x)
                y = int(self.selection_rect[1] * scale_y)
                w = int(self.selection_rect[2] * scale_x)
                h = int(self.selection_rect[3] * scale_y)
                
                # Рисуем прямоугольник выделения
                painter.drawRect(x, y, w, h)
                painter.end()
            
            self.image_label.setPixmap(pixmap)
            
            if self.hist_initialized:
                self.update_histogram()
            else:
                self.init_histogram()
    
    def slider_pressed(self):
        """Слайдер нажат - ничего не делаем"""
        pass
    
    def slider_released(self):
        """Слайдер отпущен - применяем преобразования"""
        self.brightness_value = self.brightness_slider.value()
        self.contrast_value = self.contrast_slider.value()
        self.brightness_value_label.setText(str(self.brightness_value))
        self.contrast_value_label.setText(str(self.contrast_value))
        self.apply_all_transforms()
    
    def binary_threshold_changed(self, value):
        """Обработчик изменения порога бинаризации"""
        self.binary_threshold = value
        self.binary_threshold_label.setText(str(value))
    
    def apply_all_transforms(self):
        """Применить все активные преобразования к изображению"""
        if self.original_array is None:
            return
        
        # Определяем область для обработки
        if self.selection_active and self.selection_rect:
            x, y, w, h = self.selection_rect
            roi = self.original_array[y:y+h, x:x+w].copy().astype(np.float32)
            
            # Применяем контрастность
            if self.contrast_value != 100:
                contrast = self.contrast_value / 100.0
                mean_r = np.mean(roi[:,:,0])
                mean_g = np.mean(roi[:,:,1])
                mean_b = np.mean(roi[:,:,2])
                
                roi[:,:,0] = contrast * (roi[:,:,0] - mean_r) + mean_r
                roi[:,:,1] = contrast * (roi[:,:,1] - mean_g) + mean_g
                roi[:,:,2] = contrast * (roi[:,:,2] - mean_b) + mean_b
            
            # Применяем яркость
            if self.brightness_value != 0:
                roi += self.brightness_value
            
            # Обрезаем значения и конвертируем обратно
            roi = np.clip(roi, 0, 255).astype(np.uint8)
            
            # Обновляем только выделенную область
            self.current_array = self.original_array.copy()
            self.current_array[y:y+h, x:x+w] = roi
        else:
            # Обрабатываем все изображение
            self.current_array = self.original_array.copy().astype(np.float32)
            
            # Применяем контрастность
            if self.contrast_value != 100:
                contrast = self.contrast_value / 100.0
                mean_r = np.mean(self.current_array[:,:,0])
                mean_g = np.mean(self.current_array[:,:,1])
                mean_b = np.mean(self.current_array[:,:,2])
                
                self.current_array[:,:,0] = contrast * (self.current_array[:,:,0] - mean_r) + mean_r
                self.current_array[:,:,1] = contrast * (self.current_array[:,:,1] - mean_g) + mean_g
                self.current_array[:,:,2] = contrast * (self.current_array[:,:,2] - mean_b) + mean_b
            
            # Применяем яркость
            if self.brightness_value != 0:
                self.current_array += self.brightness_value
            
            # Обрезаем значения и конвертируем обратно
            self.current_array = np.clip(self.current_array, 0, 255).astype(np.uint8)
        
        self.update_display()
    
    def apply_binarization(self):
        """Применяет бинаризацию"""
        if self.current_array is None:
            return
        
        method = self.binary_combo.currentText()
        
        # Определяем область для обработки
        if self.selection_active and self.selection_rect:
            x, y, w, h = self.selection_rect
            roi = self.current_array[y:y+h, x:x+w].copy()
            
            # Преобразуем в grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            
            if method == 'Фиксированный порог':
                threshold = self.binary_threshold_slider.value()
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            elif method == 'Адаптивный порог':
                # Быстрая адаптивная бинаризация
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
            else:  # Метод Оцу
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Преобразуем обратно в RGB
            binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            
            # Обновляем только выделенную область
            self.current_array[y:y+h, x:x+w] = binary_rgb
        else:
            # Обрабатываем все изображение
            work_array = self.current_array.copy()
            
            # Преобразуем в grayscale
            gray = cv2.cvtColor(work_array, cv2.COLOR_RGB2GRAY)
            
            if method == 'Фиксированный порог':
                threshold = self.binary_threshold_slider.value()
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            elif method == 'Адаптивный порог':
                # Быстрая адаптивная бинаризация
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
            else:  # Метод Оцу
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Преобразуем обратно в RGB
            self.current_array = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        self.update_display()
    
    def apply_grayscale(self):
        """Применяет оттенки серого"""
        if self.current_array is None:
            return
        
        # Определяем область для обработки
        if self.selection_active and self.selection_rect:
            x, y, w, h = self.selection_rect
            roi = self.current_array[y:y+h, x:x+w].copy()
            
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            # Обновляем только выделенную область
            self.current_array[y:y+h, x:x+w] = gray_rgb
        else:
            # Обрабатываем все изображение
            gray = cv2.cvtColor(self.current_array, cv2.COLOR_RGB2GRAY)
            self.current_array = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        self.update_display()
    
    def apply_negative(self):
        """Применяет негатив"""
        if self.current_array is None:
            return
        
        # Определяем область для обработки
        if self.selection_active and self.selection_rect:
            x, y, w, h = self.selection_rect
            self.current_array[y:y+h, x:x+w] = 255 - self.current_array[y:y+h, x:x+w]
        else:
            # Обрабатываем все изображение
            self.current_array = 255 - self.current_array
        
        self.update_display()
    
    def reset_image(self):
        """Сбрасывает изображение к оригиналу"""
        if self.original_array is not None:
            self.current_array = self.original_array.copy()
            self.update_display()
            self.reset_sliders()
    
    def reset_sliders(self):
        """Сбрасывает слайдеры"""
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.brightness_value = 0
        self.contrast_value = 100
        self.brightness_value_label.setText('0')
        self.contrast_value_label.setText('100')
    
    def toggle_selection(self):
        """Включает/выключает режим выделения"""
        self.selection_active = not self.selection_active
        if self.selection_active:
            self.statusBar().showMessage('Режим выделения: рисуйте прямоугольник на изображении')
        else:
            self.statusBar().showMessage('')
            self.clear_selection()
    
    def clear_selection(self):
        """Очищает выделение"""
        self.selection_rect = None
        self.selection_active = False
        self.update_display()
        self.statusBar().showMessage('')
    
    def image_mouse_press(self, event):
        """Обработчик нажатия мыши на изображении"""
        if self.selection_active and self.current_array is not None:
            pos = event.pos()
            label_size = self.image_label.size()
            pixmap_size = self.image_label.pixmap().size() if self.image_label.pixmap() else label_size
            
            # Вычисляем смещение для центрированного изображения
            offset_x = (label_size.width() - pixmap_size.width()) // 2
            offset_y = (label_size.height() - pixmap_size.height()) // 2
            
            # Проверяем, что клик внутри изображения
            if (offset_x <= pos.x() < offset_x + pixmap_size.width() and
                offset_y <= pos.y() < offset_y + pixmap_size.height()):
                
                # Пересчитываем координаты в систему изображения
                img_x = int((pos.x() - offset_x) * self.current_array.shape[1] / pixmap_size.width())
                img_y = int((pos.y() - offset_y) * self.current_array.shape[0] / pixmap_size.height())
                
                self.selection_start = (img_x, img_y)
                self.selection_rect = (img_x, img_y, 0, 0)
    
    def image_mouse_move(self, event):
        """Обработчик движения мыши на изображении"""
        if (self.selection_active and self.selection_start is not None and 
            self.current_array is not None):
            
            pos = event.pos()
            label_size = self.image_label.size()
            pixmap_size = self.image_label.pixmap().size() if self.image_label.pixmap() else label_size
            
            # Вычисляем смещение для центрированного изображения
            offset_x = (label_size.width() - pixmap_size.width()) // 2
            offset_y = (label_size.height() - pixmap_size.height()) // 2
            
            # Проверяем, что движение внутри изображения
            if (offset_x <= pos.x() < offset_x + pixmap_size.width() and
                offset_y <= pos.y() < offset_y + pixmap_size.height()):
                
                # Пересчитываем координаты в систему изображения
                img_x = int((pos.x() - offset_x) * self.current_array.shape[1] / pixmap_size.width())
                img_y = int((pos.y() - offset_y) * self.current_array.shape[0] / pixmap_size.height())
                
                start_x, start_y = self.selection_start
                width = img_x - start_x
                height = img_y - start_y
                
                # Ограничиваем выделение размерами изображения
                width = max(0, min(width, self.current_array.shape[1] - start_x))
                height = max(0, min(height, self.current_array.shape[0] - start_y))
                
                self.selection_rect = (start_x, start_y, width, height)
                self.update_display()
    
    def image_mouse_release(self, event):
        """Обработчик отпускания мыши на изображении"""
        if self.selection_active and self.selection_start is not None:
            self.selection_start = None
            if self.selection_rect[2] > 0 and self.selection_rect[3] > 0:
                self.statusBar().showMessage(f'Выделена область: {self.selection_rect[2]}x{self.selection_rect[3]} пикселей')
            else:
                self.clear_selection()
    
    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
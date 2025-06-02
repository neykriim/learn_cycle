import tkinter as tk
from tkinter import filedialog, ttk
import torch
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import time
import threading
import pandas as pd
import geopandas as gpd
import json
import torchvision.transforms as T
import torchvision.transforms.functional as F
import segmentation_models_pytorch as smp
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

matplotlib.use('TkAgg')

class FrequencyFilter(torch.nn.Module):
    def __init__(self, filter_size=2):
        """
        Args:
            filter_size: int - абсолютный размер в пикселях
            filter_type: 'highpass' или 'lowpass'
        """
        super(FrequencyFilter, self).__init__()
        self.__filter_size = filter_size
    
    def forward(self, x):
        for index in range(x.shape[0]):
            layer = x[index].clone()
            # FFT
            fft = torch.fft.fft2(layer)
            fft_shift = torch.fft.fftshift(fft)
            
            # Создаем маску
            h, w = layer.shape
            crow, ccol = h // 2, w // 2
            mask = torch.ones((h, w), device=layer.device)
            mask[crow-self.__filter_size:crow+self.__filter_size, 
                ccol-self.__filter_size:ccol+self.__filter_size] = 0
            
            # Применяем фильтр
            filtered_fft = fft_shift * mask
            x[index] = torch.fft.ifft2(torch.fft.ifftshift(filtered_fft)).real
        
        return x

class HistogramEqualizer(torch.nn.Module):
    def __init__(self, bins=256):
        super(HistogramEqualizer, self).__init__()
        self.__bins = bins
    
    def forward(self, x):
        """
        Args:
            x: Tensor shape (..., H, W) - любой размерности
        Returns:
            Equalized tensor same shape as input
        """
        
        for index in range(x.shape[0]):
            layer = x[index].clone()
            orig_shape = layer.shape
            x_flat = layer.flatten()
            
            # Вычисляем min/max если не заданы
            min_val = x_flat.min()
            max_val = x_flat.max()
            
            # Нормализуем в [0, bins-1]
            x_norm = (x_flat - min_val) * (self.__bins - 1) / (max_val - min_val + 1e-8)
            x_norm = torch.clamp(x_norm, 0, self.__bins - 1)
            
            # Вычисляем гистограмму и CDF
            hist = torch.bincount(x_norm.to(torch.long), minlength=self.__bins)
            cdf = torch.cumsum(hist, dim=0)
            cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8)
            
            # Применяем эквализацию
            equalized = cdf_normalized[x_norm.to(torch.long)]
            x[index] = equalized.reshape(orig_shape)
        
        return x

class MeanStdScaler(torch.nn.Module):
    '''
    Скалирование от среднего до плюс-минус std
    '''
    def __init__(self, mean, std):
        super(MeanStdScaler, self).__init__()
        self.__mean = mean
        self.__std = std
    
    def forward(self, obj: torch.Tensor):
        for index in range(obj.shape[0]):
            layer = obj[index].clone()
            obj[index] = (layer - self.__mean[index]) / (self.__std[index] + 1e-8)
        return obj

class IceNice:
    def __init__(self, root):
        self.root = root
        self.root.title("IceNice")
        
        # Инициализация переменных
        self.original_tensor = None
        self.processed_tensor = None
        self.current_layer = 0
        self.current_processed_layer = 0
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.dragging = False
        self.divider_pos = 0.5
        self.dragging_divider = False
        
        self.layerNames = ('one', 'two', 'strd', 'e', 'u10', 'v10', 't2m', 'sp')
        parameters = pd.read_csv('dataset_parameters.csv', index_col=0)
        parameters = parameters[parameters['layer'].isin(self.layerNames)]

        self.__transform = {
            "image": T.Compose([
                T.ToTensor(),
                FrequencyFilter(2),
                HistogramEqualizer(bins=256),
                T.Resize(1024)
            ]),
            'climate': T.Compose([
                T.ToTensor(),
                MeanStdScaler(parameters['mean'][2:].values, parameters['std'][2:].values),
                T.Resize(1024)
            ]),
            'label': T.Compose([
                T.ToTensor(),
                T.Resize(1024, interpolation=T.InterpolationMode.NEAREST)
            ]),
            'common': T.Compose([
                T.Lambda(lambda x: F.rotate(x, random.choice([0, 90, 180, 270]))),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5)
            ])
        }
        self.__device = torch.device('cuda')

        os.environ['TORCH_HOME'] = 'models'

        #self.__model = torch.load("mlruns/522018903483609275/90deaebc8b1a4d2f8d9b90756f775183/artifacts/best_vloss/data/model.pth", weights_only=False).to(self.__device)
        self.__model = torch.load("app.pth", weights_only=False).to(self.__device)
        self.__model.eval()

        with open('ice_types_dict.json', 'r', encoding='utf-8-sig') as f:
            self.__ice_dict = json.load(f)['cifer']
        #for cat in self.__ice_dict['fields'].keys():
        #    vals = [int(i) for i in self.__ice_dict['cifer'][cat].keys()]
        #    for field in self.__ice_dict['fields'][cat]:
        #        self.__ice_types[field] = self.__ice_types[field].map(dict(zip(vals, [i / (len(vals) - 1) for i in range(0, len(vals))] )))

        self.__ice_types = pd.read_csv("/home/prokofev.a@agtu.ru/Загрузки/qgis temp/Обучение моделей/dataset/ice_types.csv", index_col=0, dtype=np.int32)
        self.__maskNames = ('SA', 'FA')
        self.__maskClasses = [[87, 85, 91],
                              [8, 5, 4]]

        # Создание интерфейса
        self.create_menu()
        self.create_canvas()
        self.create_layer_buttons()
        self.create_process_button()
        #self.create_text_areas()
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Открыть", command=self.open_file)
        file_menu.add_command(label="Сохранить", command=self.save_file)
        menubar.add_cascade(label="Файл", menu=file_menu)
        self.root.config(menu=menubar)
    
    def create_canvas(self):
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Привязка событий мыши
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-2>", self.on_middle_press)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_release)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
    def create_layer_buttons(self):
        # Левая панель для слоёв исходного изображения
        self.left_button_frame = tk.Frame(self.root)
        self.left_button_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Правая панель для слоёв обработанного изображения (будет показана после обработки)
        self.right_button_frame = tk.Frame(self.root)
        
        self.layer_buttons = []
        self.processed_layer_buttons = []
        
    def create_process_button(self):
        self.process_button = tk.Button(self.root, text="Обработать", command=self.process_image)
        self.process_button.pack(side=tk.BOTTOM, pady=5)
        self.process_button.pack_forget()  # Скрываем кнопку изначально
        
    # def create_text_areas(self):
    #     # Верхний левый текст
    #     self.top_left_text = tk.Label(self.root, text="", bg="white", relief=tk.SUNKEN, padx=10, pady=5)
    #     self.top_left_text.place(relx=0, rely=0, anchor=tk.NW)
        
    #     # Верхний правый текст
    #     self.top_right_text = tk.Label(self.root, text="", bg="white", relief=tk.SUNKEN, padx=10, pady=5)
    #     self.top_right_text.place(relx=1, rely=0, anchor=tk.NE)
    
    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("GPKG files", "*.gpkg")])
        if file_path:
            self.show_loading()
            self.root.update()
            
            # В реальной реализации здесь будет ваш код для чтения .gpkg файла
            # Для примера создадим случайный тензор и названия слоёв
            threading.Thread(target=self.load_file, args=(file_path,)).start()

    def save_file(self):
        pass
    
    def load_file(self, file_path):

        data = gpd.read_file(file_path).sort_values(by=['point_id'])
        data = data.rename(columns={"vv": "one", "hh": "one", "vh": "two", "hv": "two"})

        #layer_names = ['one', 'two', 'ssrd', 'strd', 'e', 'u10', 'v10', 't2m', 'sst', 'sp', 'rsn', 'sd', 'lsm']

        for layer in self.layerNames:
            if data[layer].isna().any(axis=0):
                mean = data[layer][data[layer].notna()].mean()
                mean = 0 if mean is np.nan else mean
                data[layer] = data[layer].fillna(mean)

        image_channels = []
        for layer in self.layerNames:
            image_channels.append(np.reshape(data[layer].to_numpy(dtype=np.float32), (1000, 1000), order="F"))
        image = np.stack(image_channels, axis=-1)

        self.original_tensor = torch.cat((self.__transform['image'](image[:,:,:2]), self.__transform['climate'](image[:,:,2:])))
        self.original_tensor = self.__transform['common'](self.original_tensor)

        # Обновляем интерфейс в основном потоке
        self.root.after(0, self.update_after_load)
    
    def update_after_load(self):
        #self.layer_names = layer_names
        self.current_layer = 0
        
        # Очищаем старые кнопки
        for btn in self.layer_buttons:
            btn.pack_forget()
        self.layer_buttons = []
        
        # Создаем новые кнопки для слоёв
        for i, name in enumerate(self.layerNames):
            btn = tk.Button(self.left_button_frame, text=name, 
                           command=lambda idx=i: self.switch_layer(idx))
            btn.pack(fill=tk.X, padx=2, pady=2)
            self.layer_buttons.append(btn)
        
        # Показываем первый слой
        self.show_current_layer()
        self.process_button.pack(side=tk.BOTTOM, pady=5)
    
    def show_loading(self):
        self.canvas.delete("all")
        self.canvas.create_text(self.canvas.winfo_width()/2, self.canvas.winfo_height()/2, 
                               text="Загрузка...", font=("Arial", 24), tags="loading")
    
    def show_current_layer(self):
        if self.original_tensor is None:
            return
            
        self.canvas.delete("all")
        
        # Получаем текущий слой и нормализуем его для отображения
        layer = self.original_tensor[self.current_layer].numpy()
        layer = (layer - layer.min()) / (layer.max() - layer.min()) * 255
        layer = layer.astype(np.uint8)
        
        # Создаем изображение из массива
        self.original_image = Image.fromarray(layer)
        self.update_display()
    
    def show_processed_layer(self):
        if self.processed_tensor is None:
            return
        
        labels = self.processed_tensor[self.current_processed_layer]

        unique_classes = np.unique(labels)

        np.random.seed(0)
        class_colors = {cls: np.random.rand(3) for cls in unique_classes}

        cmap = mcolors.ListedColormap([class_colors[cls] for cls in unique_classes])

        sorted_classes = np.sort(unique_classes)
        bounds = (sorted_classes[:-1] + sorted_classes[1:]) / 2  # Средние точки между классами
        bounds = np.concatenate(([sorted_classes[0] - 1], bounds, [sorted_classes[-1] + 1]))  # Добавляем крайние значения

        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        fig, ax = plt.subplots()#figsize=(8, 6))
        im = ax.imshow(labels, cmap=cmap, norm=norm)
        ax.axis('off')

        cbar = plt.colorbar(im, ax=ax, ticks=unique_classes)
        cbar.set_label('Классы')

        # with open('ice_types_dict.json', 'r', encoding='utf-8-sig') as f:
        #     ice_dict = json.load(f)['cifer']
        layers = self.__maskNames
        layers = [i[0] for i in layers]
        #print(self.__ice_dict[layers[self.current_processed_layer]])
        cbar.ax.set_yticklabels([self.__ice_dict[layers[self.current_processed_layer]][str(i)] for i in [self.__maskClasses[self.current_processed_layer][i] for i in unique_classes]])
        fig.canvas.manager.window.attributes('-topmost', 1)
        fig.canvas.manager.window.attributes('-topmost', 0)
        fig.show()

        # # Получаем текущий слой и нормализуем его для отображения
        # layer = self.processed_tensor[self.current_processed_layer].numpy()#.astype(np.int32)
        # #plt.imshow(layer)
        # #plt.show()
        # indexes = np.unique(layer)
        # #print(indexes)
        # keys = [i for i in self.__ice_dict['cifer'].keys()]
        # values = [self.__ice_dict['cifer'][keys[self.current_processed_layer]][key] for key in self.__ice_dict['cifer'][keys[self.current_processed_layer]].keys() if not key in ('-9', '99')]

        # self.set_top_right_text("\n".join([values[i] for i in indexes]))
        # layer = (layer - layer.min()) / (layer.max() - layer.min()) * 255
        # #layer = layer.astype(np.uint8)
        
        # # Создаем изображение из массива
        # self.processed_image = Image.fromarray(layer)
        # self.update_display()
    
    def update_display(self):
        if not hasattr(self, 'original_image'):
            return
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Масштабируем оригинальное изображение
        orig_size = (
            int(self.original_image.width * self.scale), 
            int(self.original_image.height * self.scale)
        )
        orig_img = self.original_image.resize(orig_size, Image.Resampling.LANCZOS)
        
        # Позиция оригинального изображения
        orig_x = (canvas_width * self.divider_pos) / 2 + self.offset_x
        orig_y = canvas_height / 2 + self.offset_y
        
        # Преобразуем в формат Tkinter
        self.tk_orig_img = ImageTk.PhotoImage(orig_img)
        
        # Очищаем холст и рисуем оригинальное изображение
        self.canvas.delete("all")
        self.canvas.create_image(orig_x, orig_y, image=self.tk_orig_img, anchor=tk.CENTER)
        
        # Если есть обработанное изображение
        if hasattr(self, 'processed_image'):
            pass
            # Масштабируем обработанное изображение
            #proc_size = (
            #    int(self.processed_image.width * self.scale), 
            #    int(self.processed_image.height * self.scale)
            #)
            #proc_img = self.processed_image.resize(proc_size, Image.Resampling.LANCZOS)
            
            # Позиция обработанного изображения
            #proc_x = canvas_width * self.divider_pos + (canvas_width * (1 - self.divider_pos)) / 2 + self.offset_x
            #proc_y = canvas_height / 2 + self.offset_y
            
            # Преобразуем в формат Tkinter
            #self.tk_proc_img = ImageTk.PhotoImage(proc_img)
            
            # Рисуем обработанное изображение
            #self.canvas.create_image(orig_x, orig_y, image=self.tk_proc_img, anchor=tk.CENTER)
            
            # Рисуем разделитель
            #divider_x = canvas_width * self.divider_pos
            #self.canvas.create_line(divider_x, 0, divider_x, canvas_height, fill="red", width=2, tags="divider")
            
            # Делаем разделитель перетаскиваемым
            #self.canvas.tag_bind("divider", "<ButtonPress-1>", self.start_divider_drag)
            #self.canvas.tag_bind("divider", "<B1-Motion>", self.drag_divider)
            #self.canvas.tag_bind("divider", "<ButtonRelease-1>", self.stop_divider_drag)
    
    def switch_layer(self, layer_idx):
        self.current_layer = layer_idx
        self.show_current_layer()
    
    def switch_processed_layer(self, layer_idx):
        self.current_processed_layer = layer_idx
        self.show_processed_layer()
    
    def process_image(self):
        if self.original_tensor is None:
            return
            
        self.show_loading()
        self.process_button.pack_forget()
        self.root.update()
        
        # В реальной реализации здесь будет вызов вашей PyTorch модели
        # Для примера создадим случайный тензор 4x1000x1000
        threading.Thread(target=self.run_processing).start()
    
    def run_processing(self):
        self.original_tensor = self.original_tensor.to(self.__device)
        #self.__maskNames
        self.__model.eval()
        with torch.no_grad():
            prediction = self.__model(self.original_tensor[None, :, :, :]).cpu()[0]
            self.original_tensor = self.original_tensor.cpu()

        self.processed_tensor = []
        for i in range(len(self.__maskClasses)):
            m = 0
            for j in range(i):
                m += len(self.__maskClasses[j])
            labels = prediction[m : m + len(self.__maskClasses[i])]
            labels = torch.nn.functional.softmax(labels, dim=0)
            labels = torch.argmax(labels, dim=0)

            self.processed_tensor.append(labels)

        # Обновляем интерфейс в основном потоке
        self.root.after(0, self.update_after_processing)
    
    def update_after_processing(self):
        self.current_processed_layer = 0
        
        # Очищаем старые кнопки
        for btn in self.processed_layer_buttons:
            btn.pack_forget()
        self.processed_layer_buttons = []
        
        # Показываем правую панель с кнопками
        self.right_button_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Создаем новые кнопки для слоёв
        for i, name in enumerate(self.__maskNames):
            btn = tk.Button(self.right_button_frame, text=name, 
                           command=lambda idx=i: self.switch_processed_layer(idx))
            btn.pack(fill=tk.X, padx=2, pady=2)
            self.processed_layer_buttons.append(btn)
        
        # Показываем первый слой
        self.show_processed_layer()
    
    def on_mouse_wheel(self, event):
        # Масштабирование при прокрутке колеса мыши
        scale_factor = 1.1 if event.delta > 0 else 0.9
        self.scale *= scale_factor
        
        # Ограничиваем масштаб
        self.scale = max(0.1, min(self.scale, 10.0))
        
        self.update_display()
    
    def on_middle_press(self, event):
        # Начало перемещения
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.dragging = True
    
    def on_middle_drag(self, event):
        if self.dragging:
            # Вычисляем смещение
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            
            self.offset_x += dx
            self.offset_y += dy
            
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            
            self.update_display()
    
    def on_middle_release(self, event):
        self.dragging = False
    
    def start_divider_drag(self, event):
        self.dragging_divider = True
    
    def drag_divider(self, event):
        if self.dragging_divider:
            canvas_width = self.canvas.winfo_width()
            new_pos = event.x / canvas_width
            
            # Ограничиваем позицию разделителя
            self.divider_pos = max(0.2, min(0.8, new_pos))
            self.update_display()
    
    def stop_divider_drag(self, event):
        self.dragging_divider = False
    
    def on_canvas_resize(self, event):
        self.update_display()
    
    def set_top_left_text(self, text):
        self.top_left_text.config(text=text)
    
    def set_top_right_text(self, text):
        self.top_right_text.config(text=text)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    app = IceNice(root)
    root.mainloop()
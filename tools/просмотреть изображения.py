import tkinter as tk
from tkinter import ttk
import geopandas as gpd
import numpy as np
from PIL import Image, ImageTk
import os
import re
#from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

class GPKGViewerApp:
    MAX_CLASSES = 10  # Максимальное количество классов
    IMAGE_SIZE = 500  # Фиксированный размер изображения
    
    def __init__(self, root, folder_path):
        self.root = root
        self.folder_path = folder_path
        self.root.title("GPKG Viewer")
        self.setup_ui()
        self.load_catalogs()
        
    def setup_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.root.geometry("1200x700")
        
        # Main paned window for resizable panels
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=8)
        self.paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - catalogs list (resizable)
        self.setup_left_panel()
        
        # Right panel - content display
        self.setup_right_panel()
        
        # Initialize state variables
        self.current_catalog = None
        self.current_image_index = 1
        self.max_image_index = 1
        self.catalogs = {}
        self.tab_data = {}  # Словарь для хранения данных по вкладкам
        self.current_file_data = None  # Храним данные текущего файла
        
    def setup_left_panel(self):
        """Настройка левой панели с каталогами"""
        self.left_frame = tk.Frame(self.paned, width=250, bg="#f0f0f0")
        self.paned.add(self.left_frame, minsize=150, stretch="always")
        
        tk.Label(self.left_frame, text="Каталоги", bg="#f0f0f0", 
                font=('Arial', 10, 'bold')).pack(pady=(5, 0))
        
        self.catalog_listbox = tk.Listbox(
            self.left_frame,
            selectbackground="#4a98f7",
            selectforeground="white",
            font=('Arial', 10),
            relief=tk.FLAT,
            highlightthickness=0,
            width=30
        )
        scrollbar = tk.Scrollbar(self.left_frame, orient=tk.VERTICAL)
        scrollbar.config(command=self.catalog_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.catalog_listbox.config(yscrollcommand=scrollbar.set)
        self.catalog_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.catalog_listbox.bind('<<ListboxSelect>>', self.on_catalog_select)
    
    def setup_right_panel(self):
        """Настройка правой панели с контентом"""
        self.right_frame = tk.Frame(self.paned)
        self.paned.add(self.right_frame, minsize=400, stretch="always")
        
        # Navigation controls
        self.setup_navigation()
        
        # Content display
        self.setup_content_display()
        
    def setup_navigation(self):
        """Настройка панели навигации"""
        nav_frame = tk.Frame(self.right_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Button(nav_frame, text="◄", command=self.prev_image,
                width=3).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(nav_frame, text="►", command=self.next_image,
                width=3).pack(side=tk.LEFT)
        
        tk.Label(nav_frame, text="Изображение:").pack(side=tk.LEFT, padx=(10, 0))
        
        self.image_index_entry = tk.Entry(nav_frame, width=5)
        self.image_index_entry.pack(side=tk.LEFT)
        self.image_index_entry.bind("<Return>", self.on_image_index_change)
        
        self.image_count_label = tk.Label(nav_frame, text="/ 0")
        self.image_count_label.pack(side=tk.LEFT)
        
        # Add classification checkbox
        self.classify_var = tk.BooleanVar()
        self.classify_check = tk.Checkbutton(
            nav_frame, 
            text="Классифицировать", 
            variable=self.classify_var,
            command=self.toggle_classification
        )
        self.classify_check.pack(side=tk.LEFT, padx=(20, 0))
    
    def setup_content_display(self):
        """Настройка области отображения контента"""
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Loading indicator
        self.loading_label = tk.Label(self.right_frame, text="Загрузка...", 
                                    font=('Arial', 12))
        
    def toggle_classification(self):
        """Переключение режима классификации"""
        if self.current_file_data is not None:
            self.update_display_from_loaded_data()
    
    def load_catalogs(self):
        """Загрузка списка каталогов из папки"""
        catalog_files = {}
        
        for file in os.listdir(self.folder_path):
            if file.endswith(".gpkg"):
                match = re.match(r"^(.+)_(\d+)\.gpkg$", file)
                if match:
                    catalog_name = match.group(1)
                    image_num = int(match.group(2))
                    
                    if catalog_name not in catalog_files:
                        catalog_files[catalog_name] = []
                    catalog_files[catalog_name].append(image_num)
        
        for catalog_name in sorted(catalog_files.keys()):
            image_numbers = sorted(catalog_files[catalog_name])
            self.catalogs[catalog_name] = {
                'min_index': min(image_numbers),
                'max_index': max(image_numbers),
                'files': {num: f"{catalog_name}_{num}.gpkg" for num in image_numbers}
            }
            self.catalog_listbox.insert(tk.END, catalog_name)
    
    def show_loading(self):
        """Показать индикатор загрузки"""
        self.notebook.pack_forget()
        self.loading_label.pack(fill=tk.BOTH, expand=True)
        self.root.update_idletasks()
    
    def hide_loading(self):
        """Скрыть индикатор загрузки"""
        self.loading_label.pack_forget()
        self.notebook.pack(fill=tk.BOTH, expand=True)
    
    def on_catalog_select(self, event):
        """Обработчик выбора каталога"""
        if not self.catalog_listbox.curselection():
            return
            
        selected_catalog = self.catalog_listbox.get(self.catalog_listbox.curselection())
        self.current_catalog = selected_catalog
        self.current_image_index = self.catalogs[selected_catalog]['min_index']
        self.max_image_index = self.catalogs[selected_catalog]['max_index']
        
        self.image_index_entry.delete(0, tk.END)
        self.image_index_entry.insert(0, str(self.current_image_index))
        self.image_count_label.config(text=f"/ {self.max_image_index}")
        
        self.show_loading()
        self.root.after(100, self.load_current_image)
    
    def load_current_image(self):
        """Загрузка текущего изображения"""
        if not self.current_catalog:
            return
            
        filename = self.catalogs[self.current_catalog]['files'].get(self.current_image_index)
        if not filename:
            return
            
        csv_path = os.path.join(self.folder_path, filename)
        
        try:
            # Читаем данные с диска только при смене файла
            csv_data = gpd.read_file(csv_path).sort_values(by=["point_id"])
            self.current_file_data = {
                'data': csv_data,
                'fields': [col for col in csv_data.columns if col not in ['point_id', 'geometry']]
            }
            
            self.update_display_from_loaded_data()
        except Exception as e:
            print(f"Error loading file: {e}")
            error_frame = tk.Frame(self.notebook)
            tk.Label(error_frame, text=f"Ошибка загрузки: {str(e)}").pack()
            self.notebook.add(error_frame, text="Ошибка")
        finally:
            self.hide_loading()
    
    def update_display_from_loaded_data(self):
        """Обновление отображения из уже загруженных данных"""
        if self.current_file_data is None:
            return
            
        # Clear previous content
        for child in self.notebook.winfo_children():
            child.destroy()
        
        self.tab_data = {}  # Очищаем предыдущие данные
        
        # Закрываем все предыдущие фигуры matplotlib
        plt.close('all')
        
        for field in self.current_file_data['fields']:
            self.create_image_tab(field)
    
    def create_image_tab(self, field):
        """Создание вкладки с изображением"""
        frame = tk.Frame(self.notebook)
        self.notebook.add(frame, text=field)
        
        try:
            # Получаем данные из уже загруженного файла
            image_data = np.reshape(
                self.current_file_data['data'][field].to_numpy(dtype=np.float32),
                (1000, 1000), 
                order="F"
            )
            
            self.tab_data[field] = {
                'original_data': image_data,
                'frame': frame,
                'photo': None,
                'label': None,
                'canvas': None
            }
            
            # Создаем отображение
            if self.classify_var.get():
                self.show_classified_image(frame, image_data, field)
            else:
                self.show_normal_image(frame, image_data, field)
            
        except Exception as e:
            tk.Label(frame, text=f"Ошибка отображения {field}: {str(e)}").pack()
    
    def show_normal_image(self, frame, image_data, field):
        """Отображение обычного изображения фиксированного размера"""
        # Normalize data
        min_val = np.min(image_data)
        max_val = np.max(image_data)
        if max_val > min_val:
            normalized = (image_data - min_val) / (max_val - min_val) * 255
        else:
            normalized = np.zeros_like(image_data)
        
        # Create PIL image и ресайзим до фиксированного размера
        img = Image.fromarray(normalized.astype(np.uint8))
        img = img.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        # Create label and center it
        label = tk.Label(frame, image=photo)
        label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Store references
        self.tab_data[field]['photo'] = photo
        self.tab_data[field]['label'] = label
        
        # Bind mouse wheel for zoom
        frame.bind("<MouseWheel>", lambda e: self.on_mouse_wheel(e, field))
    
    def show_classified_image(self, frame, image_data, field):
        """Отображение классифицированного изображения с легендой"""
        # Prepare figure
        fig, (ax_img, ax_legend) = plt.subplots(
            1, 2, 
            gridspec_kw={'width_ratios': [4, 1]},
            figsize=(8, 6)
        )
        fig.subplots_adjust(wspace=0.1)
        
        # Classify data
        unique_vals = np.unique(image_data)
        num_unique = len(unique_vals)
        
        if num_unique <= self.MAX_CLASSES:
            # Classification by unique values
            classes = np.digitize(image_data, unique_vals) - 1
            class_labels = [f"{val:.2f}" for val in unique_vals]
        else:
            # Classification by quantiles
            quantiles = np.linspace(0, 1, self.MAX_CLASSES + 1)
            thresholds = np.quantile(image_data, quantiles)
            classes = np.digitize(image_data, thresholds) - 1
            class_labels = [f"{thresholds[i]:.2f}-{thresholds[i+1]:.2f}" 
                          for i in range(self.MAX_CLASSES)]
        
        # Create colormap
        cmap = plt.get_cmap('viridis', len(class_labels))
        
        # Display classified image
        im = ax_img.imshow(classes, cmap=cmap, vmin=0, vmax=len(class_labels)-1)
        ax_img.axis('off')
        
        # Create legend
        patches = [mpatches.Patch(color=cmap(i), label=class_labels[i]) 
                  for i in range(len(class_labels))]
        ax_legend.legend(
            handles=patches, 
            loc='center', 
            frameon=False,
            title="Классы"
        )
        ax_legend.axis('off')
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store references
        self.tab_data[field]['figure'] = fig
        self.tab_data[field]['canvas'] = canvas
        
        # Bind mouse wheel for zoom
        canvas.get_tk_widget().bind("<MouseWheel>", lambda e: self.on_mouse_wheel(e, field))
    
    def on_mouse_wheel(self, event, field):
        """Обработчик колесика мыши для масштабирования"""
        # Здесь можно реализовать масштабирование по колесику мыши
        # В текущей версии просто игнорируем, так как изображения фиксированного размера
        pass
    
    def prev_image(self):
        """Перейти к предыдущему изображению"""
        if (self.current_catalog and 
            self.current_image_index > self.catalogs[self.current_catalog]['min_index']):
            self.current_image_index -= 1
            self.update_image_display()
    
    def next_image(self):
        """Перейти к следующему изображению"""
        if (self.current_catalog and 
            self.current_image_index < self.catalogs[self.current_catalog]['max_index']):
            self.current_image_index += 1
            self.update_image_display()
    
    def on_image_index_change(self, event):
        """Обработчик изменения номера изображения"""
        if not self.current_catalog:
            return
            
        try:
            new_index = int(self.image_index_entry.get())
            min_idx = self.catalogs[self.current_catalog]['min_index']
            max_idx = self.catalogs[self.current_catalog]['max_index']
            
            if min_idx <= new_index <= max_idx:
                self.current_image_index = new_index
                self.update_image_display()
            else:
                self.reset_image_index()
        except ValueError:
            self.reset_image_index()
    
    def reset_image_index(self):
        """Сброс номера изображения на текущее значение"""
        self.image_index_entry.delete(0, tk.END)
        self.image_index_entry.insert(0, str(self.current_image_index))
    
    def update_image_display(self):
        """Обновление отображения изображения"""
        self.image_index_entry.delete(0, tk.END)
        self.image_index_entry.insert(0, str(self.current_image_index))
        self.show_loading()
        self.root.after(100, self.load_current_image)

if __name__ == "__main__":
    root = tk.Tk()
    folder_path = os.path.dirname(os.path.abspath(__file__))
    app = GPKGViewerApp(root, folder_path)
    root.mainloop()
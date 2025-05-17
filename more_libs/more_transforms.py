import torch

class RandomRotation(torch.nn.Module):
    '''
    Случайный поворот изображения
    '''
    def __init__(self, counter=3,p=0.75):
        super(RandomRotation, self).__init__()
        self.__counter = counter
        self.__currentCounter = counter
        self.__result = 0
        self.__angle = 0
        self.__p = p
    
    def forward(self, obj: torch.Tensor):
        if self.__currentCounter == self.__counter:
            self.__result = torch.rand(1)
            self.__angle = random.choice([90, 180, 270])

        self.__currentCounter -= 1

        if self.__currentCounter == 0:
            self.__currentCounter = self.__counter

        if self.__result < self.__p:
            return F.rotate(obj, self.__angle)
        return obj

class MinMaxScaler(torch.nn.Module):
    '''
    Скалирование от минимума до максимума
    '''
    def __init__(self, minimum, maximum):
        super(MinMaxScaler, self).__init__()
        self.__min = minimum
        self.__max = maximum
    
    def forward(self, obj: torch.Tensor):
        for index in range(obj.shape[0]):
            layer = obj[index].clone()
            obj[index] = (layer - self.__min[index]) / (self.__max[index] - self.__min[index] + 1e-8)
        return obj

class HistogramEqualizer(torch.nn.Module):
    '''
    Выравнивание гистограммы
    '''
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

class FrequencyFilter(torch.nn.Module):
    '''
    Частотная фильтрация
    '''
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
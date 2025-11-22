import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- КОНФИГУРАЦИЯ ---
plt.style.use('seaborn-v0_8-whitegrid')  # Светлый, чистый стиль для презентаций
# Цвета: Красный (Сложная/Опасная) и Синий (Простая/Спокойная)
COLOR_COMPLEX = '#e74c3c'
COLOR_SIMPLE = '#3498db'

# Пороги
THR_MED = 0.2
THR_LARGE = 0.6


def get_demo_data():
    """Генерация данных (если нет файлов)"""
    dims = 4096
    # Простая: куча нулей, редкие пики
    simple = np.random.exponential(scale=0.05, size=dims)
    simple[0:5] = 2.5  # Пара мощных нейронов

    # Сложная: длинный "хвост", много средних значений
    complex_ = np.random.exponential(scale=0.2, size=dims)
    return complex_, simple


def plot_comparative_histograms(vec_complex, vec_simple):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Подготовка данных (берем модуль, так как важна сила активации, а не знак)
    v_c = np.abs(vec_complex)
    v_s = np.abs(vec_simple)

    # --- ГРАФИК 1: Детальная Гистограмма (Log Scale) ---
    # bins - разбиваем диапазон от 0 до макс. значения на 50 полосок
    max_val = max(v_c.max(), v_s.max())
    bins = np.linspace(0, max_val, 50)

    # Рисуем "Сложную"
    ax1.hist(v_c, bins=bins, alpha=0.6, color=COLOR_COMPLEX, label='Сложная (Взрыв)', log=True)
    # Рисуем "Простую" поверх
    ax1.hist(v_s, bins=bins, alpha=0.6, color=COLOR_SIMPLE, label='Простая (Красный)', log=True)

    ax1.set_title('Распределение силы активаций (Log Scale)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Сила активации нейрона', fontsize=12)
    ax1.set_ylabel('Количество нейронов (логарифм)', fontsize=12)
    ax1.legend(fontsize=11)

    # Добавляем пояснение прямо на график
    ax1.text(0.5, 0.5, 'У простых концепций\nбыстрый спад', transform=ax1.transAxes, fontsize=10, color=COLOR_SIMPLE,
             alpha=0.8)
    ax1.text(0.5, 0.8, 'У сложных концепций\n"тяжелый хвост"', transform=ax1.transAxes, fontsize=10,
             color=COLOR_COMPLEX, alpha=0.8)

    # --- ГРАФИК 2: Категории (Слабые / Средние / Сильные) ---
    # Считаем количество в группах
    def count_groups(vec):
        low = (vec < THR_MED).sum()
        med = ((vec >= THR_MED) & (vec < THR_LARGE)).sum()
        high = (vec >= THR_LARGE).sum()
        return [low, med, high]

    counts_c = count_groups(v_c)
    counts_s = count_groups(v_s)

    labels = ['Шум (<0.2)', 'Средние (0.2-0.6)', 'Важные (>0.6)']
    x = np.arange(len(labels))
    width = 0.35

    # Рисуем столбцы
    rects1 = ax2.bar(x - width / 2, counts_c, width, label='Сложная', color=COLOR_COMPLEX, alpha=0.8)
    rects2 = ax2.bar(x + width / 2, counts_s, width, label='Простая', color=COLOR_SIMPLE, alpha=0.8)

    ax2.set_title('Количество задействованных нейронов', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_yscale('log')  # Тоже логарифм, иначе маленькие столбики исчезнут
    ax2.set_ylabel('Количество (log)', fontsize=12)
    ax2.legend()

    # Подписываем значения над столбцами
    ax2.bar_label(rects1, padding=3, fmt='%d')
    ax2.bar_label(rects2, padding=3, fmt='%d')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Замени на загрузку своих файлов:
    v1 = torch.load("data/concept_persona_vector/explosion_vector.pt").numpy()
    v2 = torch.load("data/concept_persona_vector/red_color_vector.pt").numpy()

    v1, v2 = get_demo_data()
    plot_comparative_histograms(v1, v2)
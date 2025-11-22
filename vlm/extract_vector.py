import torch
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
from PIL import Image
import os
from tqdm import tqdm

# --- CONFIG ---
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
# Папка с изображениями, которые иллюстрируют нужный нам концепт
POSITIVE_DIR = "data/trait_data_extract/red_color/pos"
# Папка с изображениями, которые НЕ иллюстрируют этот концепт
NEGATIVE_DIR = "data/trait_data_extract/red_color/neg"

OUTPUT_VECTOR_FILE = "data/concept_persona_vector/red_color_vector.pt"

# Слой ЯЗЫКОВОЙ МОДЕЛИ, из которого будем извлекать вектор.
TARGET_LLM_LAYER_INDEX = 18

# Словарь для временного хранения "перехваченных" активаций из модели.
captured_activations = {}


def get_hook(name):
    """
    Фабрика функций-хуков. Создает и возвращает хук для конкретного слоя.
    Хук — это функция, которая "подсматривает" за внутренними данными модели во время ее работы.
    """

    def hook(model, input, output):
        """
        Эта функция будет вызываться автоматически при проходе данных через целевой слой.
        Она извлекает скрытое состояние (активации) и сохраняет его.
        """
        # output[0] - это основной тензор со скрытыми состояниями (hidden_state).
        # Усредняем его по всей длине последовательности токенов, чтобы получить единый вектор для всего ввода.
        captured_activations[name] = output[0].detach().cpu().mean(dim=1)

    return hook


def get_image_paths(directory):
    """Находит все изображения в указанной директории."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if
            f.lower().endswith('.png')]


def process_images_and_extract_vectors(image_paths, model, processor, hook_module_name, device):
    """
    Обрабатывает список изображений, пропускает их через модель и собирает векторы активаций.
    """
    all_vectors = []
    print(f"Обработка {len(image_paths)} изображений...")

    # Создаем пустой промпт, так как нас интересует только реакция модели на изображение, а не на текст.
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ""}]}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Проходим по каждому изображению в цикле.
    for path in tqdm(image_paths):
        try:
            image = Image.open(path).convert("RGB")
            # Готовим изображение и текст для подачи в модель.
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                model(**inputs)

            # Извлекаем вектор, который сохранил наш хук.
            vector = captured_activations.get(hook_module_name)
            if vector is not None:
                # Добавляем вектор в наш список. .squeeze() убирает лишние измерения.
                all_vectors.append(vector.squeeze())
            # Очищаем словарь для следующего изображения.
            captured_activations.clear()
        except Exception as e:
            print(f"Не удалось обработать файл {path}: {e}")

    return all_vectors


if __name__ == "__main__":
    # Определяем, на чем будем выполнять вычисления (GPU или CPU).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")

    print(f"Загрузка модели: {MODEL_NAME}...")
    # Конфигурация для квантизации модели (4-битная загрузка), чтобы она занимала меньше видеопамяти.
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME, device_map="auto", trust_remote_code=True, quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Переводим модель в режим оценки.
    model.eval()
    print("Модель успешно загружена.")

    # --- Поиск нужного слоя в языковой части модели ---
    # Имя класса для слоев-декодеров в модели Qwen2.5-VL.
    expected_layer_name_part = 'Qwen2_5_VLDecoderLayer'
    # Получаем список имен всех модулей, которые являются слоями-декодерами.
    llm_layer_names = [name for name, module in model.named_modules() if expected_layer_name_part in str(type(module))]

    # Проверяем, что слои найдены и наш индекс не выходит за пределы списка.
    if not llm_layer_names or TARGET_LLM_LAYER_INDEX >= len(llm_layer_names):
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось найти слой {TARGET_LLM_LAYER_INDEX}.")
        exit()

    # Выбираем имя слоя по нашему индексу.
    target_layer_name = llm_layer_names[TARGET_LLM_LAYER_INDEX]

    try:
        # Находим сам модуль (объект слоя) по его имени.
        target_module = dict(model.named_modules())[target_layer_name]
        # "Регистрируем" наш хук на этом слое. Теперь функция hook() будет вызываться при каждом прогоне данных.
        hook_handle = target_module.register_forward_hook(get_hook(target_layer_name))
        print(f"Установка хука на слой ЯЗЫКОВОЙ МОДЕЛИ: {target_layer_name}")
    except KeyError:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось найти модуль '{target_layer_name}'.")
        exit()

    # --- Основной процесс извлечения векторов ---
    pos_image_paths = get_image_paths(POSITIVE_DIR)
    neg_image_paths = get_image_paths(NEGATIVE_DIR)

    # Получаем списки векторов для "позитивных" и "негативных" изображений.
    pos_vectors = process_images_and_extract_vectors(pos_image_paths, model, processor, target_layer_name, device)
    neg_vectors = process_images_and_extract_vectors(neg_image_paths, model, processor, target_layer_name, device)

    # Проверяем, что мы смогли извлечь хотя бы по одному вектору каждого типа.
    if not pos_vectors or not neg_vectors:
        print("\nКРИТИЧЕСКАЯ ОШИБКА: Список векторов пуст.")
        exit()

    # --- Вычисление и сохранение вектора концепта ---
    # Усредняем все "позитивные" векторы, чтобы получить один обобщенный вектор.
    avg_pos_vector = torch.stack(pos_vectors).mean(dim=0)
    # Делаем то же самое для "негативных" векторов.
    avg_neg_vector = torch.stack(neg_vectors).mean(dim=0)
    # Вектор концепта = (средний позитивный) - (средний негативный).
    # Это направление в пространстве активаций, которое отвечает за наш концепт.
    concept_vector = avg_pos_vector - avg_neg_vector

    # Сохраняем итоговый вектор в файл.
    torch.save(concept_vector, OUTPUT_VECTOR_FILE)
    print(f"\nВектор из LLM сохранен в: {OUTPUT_VECTOR_FILE}")
    # Удаляем хук, чтобы он не мешал дальнейшей работе.
    hook_handle.remove()
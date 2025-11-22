import torch
import torch.nn as nn
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# --- CONFIG ---
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
# Путь к файлу с вектором концепции
VECTOR_FILE = "data/concept_persona_vector/explosion_vector.pt"

# Папка с тестовыми изображениями (для "руления")
IMAGES_TO_STEER_DIR = "data/images_for_steering/"

# Промпт, который будет задан модели для каждого изображения
PROMPT_FOR_STEERING = "Что случится в следующий момент?"

# Список коэффициентов "руления".
STEERING_COEFFICIENTS = [5.0]

# Индекс слоя языковой модели, к которому будем применять вектор
TARGET_LAYER_INDEX = 18

MAX_NEW_TOKENS = 128


class SteeringHook:
    """
    Класс-менеджер для хука, который изменяет активации модели "на лету" для "руления".
    """

    def __init__(self, model, layer_name, vector, alpha):
        # Находим модуль (слой) по его имени.
        self.module = dict(model.named_modules())[layer_name]
        # Вектор концепта, который будем добавлять.
        self.vector = vector
        # Коэффициент "руления" (сила и направление).
        self.alpha = alpha
        self.handle = None

    def __enter__(self):
        """Устанавливает хук при входе в блок 'with'."""
        self.handle = self.module.register_forward_hook(self.hook_fn)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Удаляет хук при выходе из блока 'with'."""
        if self.handle: self.handle.remove()

    def hook_fn(self, module, input, output):
        """
        Функция-хук, которая непосредственно изменяет скрытое состояние.
        Вызывается автоматически при проходе данных через целевой слой.
        """
        # 1. Выходные данные слоя (output) - это кортеж. Первый элемент - тензор скрытых состояний.
        hidden_state = output[0]

        # 2. Наш вектор имеет размерность [hidden_size]. Для сложения с hidden_state,
        # который имеет размерность [batch_size, sequence_length, hidden_size],
        # нужно расширить его до [1, 1, hidden_size]. Это позволит PyTorch "размножить"
        # (broadcast) его для всех токенов в последовательности и всех элементов в батче.
        vector_for_broadcast = self.vector.unsqueeze(0).unsqueeze(0)

        # 3. Добавляем наш вектор, умноженный на коэффициент alpha, к оригинальному скрытому состоянию.
        modified_hidden_state = hidden_state + self.alpha * vector_for_broadcast.to(hidden_state.device,
                                                                                    dtype=hidden_state.dtype)

        # 4. Собираем и возвращаем НОВЫЙ кортеж, заменяя только первый элемент (скрытое состояние).
        # Остальные элементы кортежа (если они есть) оставляем без изменений.
        return (modified_hidden_state,) + output[1:]


def generate_response(model, processor, image, text_prompt):
    """
    Генерирует текстовый ответ от модели для данного изображения и промпта.
    """
    # Формируем промпт в формате, понятном для модели.
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_prompt}]}
    ]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    # Запоминаем длину входной последовательности токенов.
    input_ids_len = inputs['input_ids'].shape[1]

    generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    # "Отрезаем" от сгенерированной последовательности начальную часть, которая была нашим вводом.
    response_ids = generated_ids[0][input_ids_len:]

    # Теперь декодируем только "чистый" ответ модели.
    response = processor.batch_decode([response_ids], skip_special_tokens=True)[0].strip()

    return response


def get_image_paths(directory):
    """Собирает пути ко всем изображениям в директории."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg'))]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")

    # 1. Загружаем все необходимые компоненты
    print("Загрузка моделей и данных...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME, dtype="auto", device_map="auto", trust_remote_code=True, quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if not os.path.exists(VECTOR_FILE):
        raise FileNotFoundError(f"Файл вектора {VECTOR_FILE} не найден.")
    concept_vector = torch.load(VECTOR_FILE)
    print("Модель и вектор успешно загружены.")

    # 2. Находим имя целевого слоя, как и в первом скрипте
    expected_layer_name_part = 'Qwen2_5_VLDecoderLayer'
    llm_layer_names = [name for name, module in model.named_modules() if expected_layer_name_part in str(type(module))]
    if not llm_layer_names or TARGET_LAYER_INDEX >= len(llm_layer_names):
        raise ValueError(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось найти слои с именем '{expected_layer_name_part}'.")
    target_layer_name = llm_layer_names[TARGET_LAYER_INDEX]
    print(f"Найдено {len(llm_layer_names)} слоев-декодеров. Цель: слой {TARGET_LAYER_INDEX} ('{target_layer_name}')")

    # Собираем пути к изображениям для тестирования
    image_paths = get_image_paths(IMAGES_TO_STEER_DIR)
    if not image_paths:
        raise FileNotFoundError(f"В папке {IMAGES_TO_STEER_DIR} не найдено изображений.")

    # Список для хранения результатов
    results = []

    # Создаем прогресс-бар для отслеживания общего процесса
    total_generations = len(image_paths) * len(STEERING_COEFFICIENTS)
    progress_bar = tqdm(total=total_generations, desc="Генерация ответов")

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            image_filename = os.path.basename(image_path)

            for alpha in STEERING_COEFFICIENTS:
                # Ответ при alpha=0.0 (без руления) генерируется без хука для чистоты эксперимента
                if alpha == 0.0:
                    response = generate_response(model, processor, image, PROMPT_FOR_STEERING)
                else:
                    # Для других коэффициентов создаем менеджер хука
                    hook_manager = SteeringHook(model, target_layer_name, concept_vector, alpha)
                    # Внутри блока 'with' хук активен
                    with hook_manager:
                        response = generate_response(model, processor, image, PROMPT_FOR_STEERING)
                    # При выходе из блока 'with' хук автоматически удаляется

                # Сохраняем результат в список.
                results.append({
                    "image_file": image_filename,
                    "prompt": PROMPT_FOR_STEERING,
                    "alpha": alpha,
                    "response": response
                })
                progress_bar.update(1)  # Обновляем прогресс-бар.

        except Exception as e:
            print(f"\nОшибка при обработке файла {image_path}: {e}")
            # В случае ошибки на одном изображении, обновляем прогресс-бар на все шаги для этого файла и продолжаем
            progress_bar.update(len(STEERING_COEFFICIENTS))

    progress_bar.close()

    # 5. Сохраняем все собранные результаты в один CSV-файл
    df_results = pd.DataFrame(results)
    output_filename = f"vlm_steering_results.csv"
    df_results.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"--- Результаты сохранены в {output_filename} ---")
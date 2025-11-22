import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import json
from tqdm import tqdm
import os

# --- CONFIG ---
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

# Какую черту будем контролировать
TRAIT_NAME = "humorous"  # humorous, sycophantic, или hallucinating

# Файлы для этой черты
VECTOR_FILE = f"data/trait_persona_vector/{TRAIT_NAME}_persona_vector.pt"
QUESTIONS_FILE = f"data/trait_data_extract/{TRAIT_NAME}.json"

# Слой, который мы будем контролировать (0-27)
TARGET_LAYER_INDEX = 16

# Коэффициенты "руления" (альфа), которые будем тестировать
STEERING_COEFFICIENTS = [-6.0, 0, 6.0]

# Параметры генерации
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7


class SteeringHook:
    """
    Класс-хук, который работает как контекстный менеджер для безопасного
    добавления и удаления.
    """

    def __init__(self, model, layer_name, vector, alpha):
        self.module = dict(model.named_modules())[layer_name]
        self.vector = vector
        self.alpha = alpha
        self.handle = None

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self.hook_fn)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle:
            self.handle.remove()

    def hook_fn(self, module, input, output):
        # output[0] - это тензор скрытых состояний
        hidden_state = output[0]
        # Прибавляем наш вектор, переместив его на то же устройство и приведя к тому же типу
        hidden_state += self.alpha * self.vector.to(hidden_state.device, dtype=hidden_state.dtype)


def generate_steered_response(model, tokenizer, question, hook_manager):
    """Генерирует ответ с активным хуком."""
    messages = [{"role": "user", "content": question}]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with hook_manager:  # Активируем хук только на время генерации
        outputs = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")

    # 1. Загружаем модель и токенизатор
    print(f"Загрузка модели: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", quantization_config=bnb_config
    )
    model.eval()
    print("Модель успешно загружена.")

    # 2. Загружаем вектор персоны
    print(f"Загрузка векторов из файла: {VECTOR_FILE}")
    persona_vectors_dict = torch.load(VECTOR_FILE)

    # Находим точное имя нашего целевого слоя
    layer_names = [name for name, module in model.named_modules() if 'Qwen2DecoderLayer' in str(type(module))]
    target_layer_name = layer_names[TARGET_LAYER_INDEX]

    steering_vector = persona_vectors_dict[target_layer_name]
    print(f"Вектор для слоя {TARGET_LAYER_INDEX} ('{target_layer_name}') успешно извлечен.")

    # 3. Загружаем ОЦЕНОЧНЫЙ набор вопросов (последние ...)
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    evaluation_questions = questions_data['questions'][35:]
    print(f"Загружено {len(evaluation_questions)} вопросов для оценки.")

    # 4. Основной цикл: генерируем ответы для каждого коэффициента и вопроса
    results = []
    for alpha in STEERING_COEFFICIENTS:
        print(f"\n--- Тестируем коэффициент alpha = {alpha} ---")
        hook_manager = SteeringHook(model, target_layer_name, steering_vector, alpha)

        for question in tqdm(evaluation_questions, desc=f"Генерация (alpha={alpha})"):
            response = generate_steered_response(model, tokenizer, question, hook_manager)
            results.append({
                "question": question,
                "alpha": alpha,
                "response": response
            })

    # 5. Сохраняем все результаты в один файл
    df_results = pd.DataFrame(results)
    output_filename = f"{TRAIT_NAME}_steering_results.csv"
    df_results.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n--- Все готово! Результаты сохранены в {output_filename} ---")
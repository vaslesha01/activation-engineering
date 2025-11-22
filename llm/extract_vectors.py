import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
from tqdm import tqdm
import os
from collections import defaultdict

# --- CONFIG ---
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
FILES_TO_PROCESS = [
    "humorous_responses_filtered.csv",
    "sycophantic_responses_filtered.csv",
    "hallucinating_responses_filtered.csv",
]
BATCH_SIZE = 4


# Словарь для хранения перехваченных активаций
captured_activations = {}


def get_hook(layer_name):
    def hook(model, input, output):
        # output[0] содержит скрытые состояния. .detach() отсоединяет тензор от графа вычислений.
        captured_activations[layer_name] = output[0].detach().cpu()

    return hook


def extract_activations_for_batch(batch, model, tokenizer, layer_names, device):
    """
    Извлекает и усредняет активации для батча данных.
    """
    global captured_activations

    prompts = []
    full_texts = []

    # 1. Воссоздаем полный текст, который был подан в модель
    for _, row in batch.iterrows():
        instruction = row['instruction_text']
        question = row['question']
        response = row['response']

        # Формируем промпт и полный текст точно так же, как при генерации
        full_user_content = f"{instruction}\n\n---\n\n{question}"
        messages = [{"role": "user", "content": full_user_content}]
        prompt_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        prompts.append(prompt_string)
        full_texts.append(prompt_string + response + tokenizer.eos_token)  # End Of Sequence

    # 2. Токенизация всего батча
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)

    batch_activations = defaultdict(list)

    # 3. Прогоняем модель и собираем активации
    with torch.no_grad():
        model(**inputs)

    # 4. Обрабатываем перехваченные активации
    for i in range(len(batch)):
        # Находим, где в полном тексте начинается ответ
        prompt_len = prompt_tokens.input_ids[i].ne(tokenizer.pad_token_id).sum().item()

        for name in layer_names:
            # Выделяем активации только для токенов ответа
            response_activations = captured_activations[name][i, prompt_len:]

            if response_activations.shape[0] > 0:
                # Усредняем активации по всем токенам ответа (стратегия Response avg)
                avg_activation = response_activations.mean(dim=0)
                batch_activations[name].append(avg_activation)

    # Очищаем словарь для следующего батча
    captured_activations.clear()

    return batch_activations


def process_trait_file(csv_path, model, tokenizer, layer_names, device):
    """
    Читает файл, извлекает активации, вычисляет и сохраняет вектор.
    """
    print(f"\n--- Обрабатываем черту из файла: {csv_path} ---")

    if not os.path.exists(csv_path):
        print(f"  Ошибка: Файл {csv_path} не найден.")
        return

    df = pd.read_csv(csv_path)

    # Разделяем на позитивные и негативные примеры
    df_pos = df[df['instruction_type'] == 'positive'].reset_index(drop=True)
    df_neg = df[df['instruction_type'] == 'negative'].reset_index(drop=True)

    print(f"  Найдено {len(df_pos)} позитивных и {len(df_neg)} негативных примеров.")

    all_pos_activations = defaultdict(list)
    all_neg_activations = defaultdict(list)

    # Обработка позитивных примеров
    print("  Извлекаем активации для позитивных примеров...")
    for i in tqdm(range(0, len(df_pos), BATCH_SIZE)):
        batch_df = df_pos.iloc[i:i + BATCH_SIZE]
        batch_activations = extract_activations_for_batch(batch_df, model, tokenizer, layer_names, device)
        for name, activations in batch_activations.items():
            all_pos_activations[name].extend(activations)

    # Обработка негативных примеров
    print("  Извлекаем активации для негативных примеров...")
    for i in tqdm(range(0, len(df_neg), BATCH_SIZE)):
        batch_df = df_neg.iloc[i:i + BATCH_SIZE]
        batch_activations = extract_activations_for_batch(batch_df, model, tokenizer, layer_names, device)
        for name, activations in batch_activations.items():
            all_neg_activations[name].extend(activations)

    # 5. Вычисляем итоговые векторы
    print("  Вычисляем векторы персоны...")
    persona_vectors = {}
    for name in layer_names:
        if all_pos_activations[name] and all_neg_activations[name]:
            # Усредняем все позитивные примеры
            avg_pos = torch.stack(all_pos_activations[name]).mean(dim=0)
            # Усредняем все негативные примеры
            avg_neg = torch.stack(all_neg_activations[name]).mean(dim=0)
            # Вычисляем разницу
            persona_vectors[name] = avg_pos - avg_neg

    # 6. Сохраняем результат
    output_filename = os.path.splitext(csv_path)[0].replace("_responses_filtered", "") + "_persona_vector.pt"
    torch.save(persona_vectors, output_filename)
    print(f"  Векторы персоны сохранены в файл: {output_filename}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")

    print(f"Загрузка модели: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Загружаем модель с 4-битной квантизацией
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config
    )
    model.eval()
    print("Модель успешно загружена.")

    # Находим все слои-декодеры в модели
    layer_names = [name for name, module in model.named_modules() if 'Qwen2DecoderLayer' in str(type(module))]
    print(f"Найдено {len(layer_names)} слоев для извлечения активаций.")

    # Регистрируем хуки на каждый слой
    hooks = []
    for name in layer_names:
        module = dict(model.named_modules())[name]
        hook = module.register_forward_hook(get_hook(name))
        hooks.append(hook)

    # Запускаем основной процесс
    for filtered_file in FILES_TO_PROCESS:
        process_trait_file(filtered_file, model, tokenizer, layer_names, device)

    # Удаляем хуки после использования, чтобы освободить ресурсы
    for hook in hooks:
        hook.remove()

    print("\n--- Извлечение всех векторов завершено! ---")
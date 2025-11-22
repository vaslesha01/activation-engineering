import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import os

# --- CONFIG ---
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
TRAIT_FILES = ["data/trait_data_extract/hallucinating.json", "data/trait_data_extract/humorous.json", "data/trait_data_extract/sycophantic.json"]
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
DO_SAMPLE = True


def generate_response(model, tokenizer, system_prompt, user_question, device):
    """
    Генерирует ответ от модели на основе системного промпта и вопроса.
    """
    # Объединяем системную инструкцию (например, "отвечай с юмором") и сам вопрос в единый текст
    full_user_content = f"{system_prompt}\n\n---\n\n{user_question}"

    messages = [{"role": "user", "content": full_user_content}]

    prompt_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_string, return_tensors="pt", padding=True).to(device)

    outputs = model.generate(
        **inputs,  # Передаем токенизированный промпт
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,  # False - модель всегда выбирает слово с самой высокой вероятностью.
        temperature=TEMPERATURE,
        # Токен-заполнитель. Чтобы сделать последовательности одинаковой длины.
        pad_token_id=tokenizer.eos_token_id  # End Of Sequence token
    )

    # Определяем длину исходного промпта, чтобы отделить его от сгенерированного ответа
    input_length = inputs['input_ids'].shape[1]

    # Декодируем (превращаем обратно в текст) только сгенерированную часть
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    # Возвращаем чистый текст ответа
    return response.strip()


def generate_contrastive_dataset(file_path: str, model, tokenizer, device):
    """
    Создает датасет контрастных ответов (позитивных и негативных) для одной черты.
    """
    print(f"\n--- Обработка файла: {file_path} ---")

    # Пытаемся загрузить JSON-файл с инструкциями и вопросами
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Ошибка: Файл {file_path} не найден.")
        return None

    # Извлекаем инструкции и первые 20 вопросов (набор для извлечения)
    instructions = data["instruction"]
    extraction_questions = data["questions"][:20]

    results = []

    progress_bar = tqdm(total=len(extraction_questions) * len(instructions), desc="Генерация ответов")

    for question in extraction_questions:
        for instr_pair in instructions:
            # Генерируем "позитивный" ответ (с проявлением черты)
            pos_response = generate_response(model, tokenizer, instr_pair['pos'], question, device)
            # Сохраняем результат
            results.append({
                "question": question,
                "instruction_type": "positive",
                "instruction_text": instr_pair['pos'],
                "response": pos_response
            })

            # Генерируем "негативный" ответ (с подавлением черты)
            neg_response = generate_response(model, tokenizer, instr_pair['neg'], question, device)
            # Сохраняем результат
            results.append({
                "question": question,
                "instruction_type": "negative",
                "instruction_text": instr_pair['neg'],
                "response": neg_response
            })

            # Обновляем progress bar
            progress_bar.update(1)

    progress_bar.close()

    df = pd.DataFrame(results)

    # Определяем имя для выходного CSV-файла и сохраняем таблицу
    output_filename = os.path.splitext(file_path)[0].replace("extract", "responses") + "_responses.csv"
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"Результаты успешно сохранены в файл: {output_filename}")
    return df

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")

    print(f"Загрузка токенизатора и модели: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Создаем конфигурацию для 4-битной квантизации
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Передаем конфигурацию квантизации при загрузке модели
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config
    )

    model.eval()
    print("Модель успешно загружена.")

    for trait_file in TRAIT_FILES:
        if os.path.exists(trait_file):
            generate_contrastive_dataset(trait_file, model, tokenizer, device)
        else:
            print(f"Предупреждение: Файл {trait_file} не найден и будет пропущен.")

    print("\n--- Все задачи выполнены! ---")
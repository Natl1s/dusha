"""
Пример использования audio_analise.py для различных сценариев

Демонстрирует типичные use cases для анализа аудио данных.
"""

from pathlib import Path
import subprocess
import sys

# Путь к скрипту
SCRIPT_PATH = Path(__file__).parent / "audio_analise.py"


def run_command(description, args):
    """Запуск команды с описанием"""
    print(f"\n{'='*70}")
    print(f"СЦЕНАРИЙ: {description}")
    print(f"{'='*70}")
    
    cmd = [sys.executable, str(SCRIPT_PATH)] + args
    print(f"Команда: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"⚠ Ошибка: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")
        return False


def main():
    """Примеры использования"""
    
    print("\n" + "="*70)
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ audio_analise.py")
    print("="*70)
    
    examples = [
        {
            "name": "Быстрый анализ (100 сэмплов)",
            "args": ["--max-samples", "100", "--save-plots"],
            "description": "Быстрая проверка работы скрипта с небольшим набором данных"
        },
        {
            "name": "Анализ с сохранением (500 сэмплов)",
            "args": ["--max-samples", "500", "--save-plots"],
            "description": "Предварительный анализ с достаточным количеством данных"
        },
        {
            "name": "Полный анализ train датасета",
            "args": ["--dataset", "combine_balanced_train_small", "--save-plots"],
            "description": "Полный анализ обучающего датасета (все сэмплы)"
        },
        {
            "name": "Анализ test датасета",
            "args": ["--dataset", "combine_balanced_test_small", "--save-plots"],
            "description": "Анализ тестового датасета для сравнения с train"
        },
    ]
    
    print("\nДоступные примеры:")
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   {example['description']}")
        print(f"   Команда: python audio_analise.py {' '.join(example['args'])}")
    
    print("\n" + "="*70)
    choice = input("\nВыберите пример для запуска (1-4) или Enter для выхода: ").strip()
    
    if not choice:
        print("Выход.")
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(examples):
            example = examples[idx]
            success = run_command(example["name"], example["args"])
            
            if success:
                print(f"\n✅ {example['name']} - выполнено успешно!")
                print(f"\nГрафики сохранены в: data_analise/visualizations/")
            else:
                print(f"\n❌ {example['name']} - выполнено с ошибками")
        else:
            print("Неверный номер примера")
    except ValueError:
        print("Неверный ввод")


if __name__ == "__main__":
    main()

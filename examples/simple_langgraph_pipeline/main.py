import subprocess
import os
import json
import tempfile
import logging
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Настройка отладки: установите переменную окружения DEBUG=1 для включения подробных логов
DEBUG = os.environ.get("DEBUG", "false").lower() in ("true", "1", "yes")
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("API_KEY", "sk-HIuZLfuAPe92nKLcbUJC4cwLRa1Zs4R5")
API_URL = "https://routerai.ru/api/v1/chat/completions"

# 1. Определяем состояние пайплайна
class PipelineState(TypedDict):
    task: str
    attempt: int
    test_script: str          # содержимое test_fs.py
    logger_script: str        # содержимое fs_logger.py
    execution_log: str        # вывод выполнения
    validation: str           # ответ нейросети-валидатора
    user_decision: Literal["retry", "abort", "approve"] | None

# 2. Узлы — обычные Python-функции
def call_ai(prompt: str) -> str:
    """Вызывает curl напрямую — 100% аналог рабочего curl."""
    logger.debug("Формирование payload для API...")
    payload = {
        "model": "deepseek/deepseek-chat-v3.1",
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": {"effort": "none"},
        "max_tokens": 1111,
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        payload_file = f.name
    
    logger.debug(f"Временный файл с payload создан: {payload_file}")
    
    try:
        logger.debug("Выполнение curl запроса...")
        result = subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                API_URL,
                "-H", f"Authorization: Bearer {API_KEY}",
                "-H", "Content-Type: application/json",
                "-d", f"@{payload_file}",
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logger.error(f"curl завершился с ошибкой: {result.stderr}")
            raise RuntimeError(f"curl failed: {result.stderr}")
        
        logger.debug("Получен ответ от API, парсинг JSON...")
        data = json.loads(result.stdout)
        content = data["choices"][0]["message"].get("content", "")
        
        if not content:
            logger.error(f"Пустой ответ от модели. Полный ответ: {data}")
            raise RuntimeError(f"Пустой ответ: {data}")
        
        logger.debug(f"Успешно получено {len(content)} символов.")
        return content
    
    finally:
        if os.path.exists(payload_file):
            os.remove(payload_file)
            logger.debug(f"Временный файл {payload_file} удалён.")

def generate_script(state: PipelineState) -> dict:
    logger.info("Генерация test_fs.py...")
    prompt = (
        "Создай test_fs.py: создаёт рядом с собой файл со случайным именем (uuid4), "
        "печатает имя, удаляет файл, печатает подтверждение. Путь — от __file__. "
        "\n\nВАЖНО: Выдай только код python"
    )
    result = {"test_script": call_ai(prompt)}
    logger.debug("Генерация test_fs.py завершена.")
    return result

def analyze_and_patch(state: PipelineState) -> dict:
    logger.info("Анализ кода и генерация fs_logger.py...")
    prompt = (
        f"Вот код:\n{state['test_script']}\n\n"
        "Найди все места работы с файловой системой. Напиши скрипт fs_logger.py, "
        "который заменяет эти участки на print-логирование. "
        "\n\nВАЖНО: Выдай ТОЛЬКО чистый код Python. Никаких пояснений, никаких markdown-обёрток (```), "
        "никаких комментариев вне кода. Только сам исполняемый код."
    )
    result = {"logger_script": call_ai(prompt)}
    logger.debug("Генерация fs_logger.py завершена.")
    return result

def execute_scripts(state: PipelineState) -> dict:
    logger.info("Сохранение и выполнение скриптов...")
    with open("test_fs.py", "w", encoding="utf-8") as f:
        f.write(state["test_script"])
    with open("fs_logger.py", "w", encoding="utf-8") as f:
        f.write(state["logger_script"])
    
    logger.debug("Запуск test_fs.py...")
    r1 = subprocess.run(["python3", "test_fs.py"], capture_output=True, text=True)
    logger.debug("Запуск fs_logger.py...")
    r2 = subprocess.run(["python3", "fs_logger.py"], capture_output=True, text=True)
    
    log = f"=== test_fs.py ===\nSTDOUT:\n{r1.stdout}\nSTDERR:\n{r1.stderr}\n"
    log += f"=== fs_logger.py ===\nSTDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}\n"
    
    logger.info("Выполнение скриптов завершено.")
    return {"execution_log": log}

def validate(state: PipelineState) -> dict:
    logger.info("Валидация результата выполнения...")
    prompt = (
        f"Проанализируй вывод:\n\n{state['execution_log']}\n\n"
        "Выполнился ли скрипт успешно? Ответь строго ОДНИМ словом: УСПЕШНО или НЕУСПЕШНО."
    )
    result = {"validation": call_ai(prompt)}
    logger.info(f"Результат валидации: {result['validation']}")
    return result

def increment_attempt(state: PipelineState) -> dict:
    new_attempt = state.get("attempt", 0) + 1
    logger.debug(f"Увеличение счётчика попыток: {new_attempt}")
    return {"attempt": new_attempt}

# 3. Условия переходов (циклы)
def route_after_validation(state: PipelineState) -> str:
    validation_lower = state["validation"].lower()
    if "успешно" in validation_lower:
        logger.debug("Маршрутизация: Успех -> ask_user")
        return "ask_user"
    if state.get("attempt", 0) >= 3:
        logger.debug("Маршрутизация: Лимит попыток исчерпан -> ask_user")
        return "ask_user"
    logger.debug("Маршрутизация: Неудача -> retry")
    return "retry"

def route_after_user(state: PipelineState) -> str:
    decision = state.get("user_decision")
    logger.debug(f"Маршрутизация по решению пользователя: {decision}")
    if decision == "retry":
        return "generate"
    if decision == "approve":
        return "end"
    return "abort"

# 4. Собираем граф
logger.info("Инициализация графа...")
graph = StateGraph(PipelineState)

graph.add_node("generate", generate_script)
graph.add_node("analyze", analyze_and_patch)
graph.add_node("execute", execute_scripts)
graph.add_node("validate", validate)
graph.add_node("increment", increment_attempt)
graph.add_node("ask_user", lambda s: None)  # breakpoint — ждём юзера

graph.add_edge(START, "generate")
graph.add_edge("generate", "analyze")
graph.add_edge("analyze", "execute")
graph.add_edge("execute", "validate")
graph.add_edge("validate", "increment")
graph.add_conditional_edges("increment", route_after_validation, {
    "retry": "generate",
    "ask_user": "ask_user"
})
graph.add_conditional_edges("ask_user", route_after_user, {
    "generate": "generate",
    "end": END,
    "abort": END
})

# 5. Human-in-the-loop: граф останавливается перед узлом ask_user
logger.info("Компиляция графа с MemorySaver и interrupt_before=['ask_user']...")
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["ask_user"]
)

# 6. Запуск и интерактивный цикл
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "run-1"}}
    initial_state = {"task": "FS test", "attempt": 0, "user_decision": None}
    
    logger.info("Запуск пайплайна...")
    
    while True:
        result = app.invoke(None, config) if "result" in locals() else app.invoke(initial_state, config)
        state = app.get_state(config)
        
        if state.next:
            logger.info(f"Граф приостановлен. Ожидаемые следующие шаги: {state.next}")
            print("\n" + "="*70)
            print("⚠️ Требуется действие пользователя")
            print(f"🔄 Текущая попытка: {state.values.get('attempt')}")
            print("="*70)
            
            # Показываем промежуточные результаты
            test_script = state.values.get('test_script', 'Не сгенерирован')
            logger_script = state.values.get('logger_script', 'Не сгенерирован')
            exec_log = state.values.get('execution_log', 'Не выполнен')
            validation = state.values.get('validation', 'Не выполнена')
            
            print("\n📄 [test_fs.py]:")
            print(test_script[:600] + ("...\n(обрезано)" if len(test_script) > 600 else ""))
            
            print("\n📄 [fs_logger.py]:")
            print(logger_script[:600] + ("...\n(обрезано)" if len(logger_script) > 600 else ""))
            
            print("\n🖥️ [Лог выполнения]:")
            print(exec_log)
            
            print("\n🤖 [Валидация нейросети]:")
            print(validation)
            
            print("\n" + "="*70)
            print("Доступные команды:")
            print("  retry   - исправить и попробовать снова")
            print("  approve - принять результат и завершить")
            print("  abort   - прервать выполнение")
            print("="*70)
            
            choice = input("Ваше решение: ").strip().lower()
            
            if choice in ["retry", "approve", "abort"]:
                logger.info(f"Пользователь выбрал: {choice}. Возобновление работы...")
                app.update_state(config, {"user_decision": choice}, as_node="ask_user")
            else:
                print("❌ Неверный ввод. Пожалуйста, введите 'retry', 'approve' или 'abort'.")
                continue
        else:
            logger.info("Пайплайн успешно завершил работу.")
            print("\n" + "="*70)
            print("✅ Пайплайн завершён")
            print(f"Итоговое состояние: {state.values}")
            print("="*70)
            break
import streamlit as st
import re
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import numpy as np

# Настройка страницы Streamlit
st.set_page_config(page_title="Обнаружение экстремистского контента",
                   layout="wide")
st.title("Обнаружение экстремистского контента")

# Боковая панель для выбора модели и параметров
st.sidebar.header("Выбор модели и параметров")
selected_model = st.sidebar.selectbox(
    "Выберите модель для анализа",
    ["distiluse", "rubert-base-cased-sentence"],
    index=0  # Установка 'distiluse' по умолчанию
)
# max_similarity = st.sidebar.slider("Порог максимального сходства", 0.0, 1.0,
#                                    0.3, 0.01)
max_similarity = 0.3

# Загрузка моделей на основе выбора
@st.cache(allow_output_mutation=True)
def load_models(model_choice):
    models = {}
    if model_choice == "distiluse":
        models['distiluse'] = SentenceTransformer(
            'distiluse-base-multilingual-cased-v2')
    elif model_choice == "rubert-base-cased-sentence":
        models['rubert-base-cased-sentence'] = SentenceTransformer(
            'DeepPavlov/rubert-base-cased-sentence')
    elif model_choice == "LaBSE":
        models['LaBSE'] = SentenceTransformer('sentence-transformers/LaBSE')
    return models


models = load_models(selected_model)

# Определение категорий и ключевых слов
CATEGORIES = {
    "terrorism": "Оправдывает терроризм или иную террористическую деятельность",
    "supremacy": "Пропагандирует исключительность, превосходство или неполноценность человека по признаку его социальной, расовой, национальной, религиозной или языковой принадлежности",
    "nazi_symbols": "Пропагандирует или демонстрирует нацистскую атрибутику или символику, сходную с нацистской, символику других экстремистских организаций",
    "hatred": "Призывает к социальной, расовой или религиозной розни, вражде",
    "minority_aggression": "Выражает агрессию или дискриминацию по отношению к меньшинствам",
    "religious_extremism": "Пропагандирует религиозный экстремизм или нетерпимость к другим религиям",
    "russophobia": "Выражает враждебность или предвзятость по отношению к русским или русской культуре",
    "act_against_russia": "Призывает к действиям против России или российских интересов",
    "migrants": "Все что связано с конфликтами мигрантов",
    "regional_problems": "Проблемы региона",
    "others": "Другие"
}

extremist_keywords = {
    "terrorism": ["терроризм", "террористический", "оправдание терроризма",
                  "террористическая деятельность", "государство-террорист"],
    "supremacy": ["превосходство", "неполноценность", "исключительность",
                  "расовое превосходство", "национальное превосходство",
                  "религиозное превосходство", "языковое превосходство",
                  "фашистское государство"],
    "nazi_symbols": ["нацистская атрибутика", "нацистская символика",
                     "свастика", "фашистский", "экстремистская символика"],
    "hatred": ["социальная рознь", "расовая рознь", "религиозная рознь",
               "вражда", "призыв к ненависти", "чтоб все сдохли",
               "смерть русне"],
    "minority_aggression": ["дискриминация меньшинств", "угнетение меньшинств",
                            "ненависть к мигрантам", "этническая чистка",
                            "расовая дискриминация", "ксенофобия"],
    "religious_extremism": ["религиозный фанатизм", "джихад",
                            "крестовый поход", "религиозная нетерпимость",
                            "сектантство", "религиозное насилие",
                            "богохульство"],
    "russophobia": ["русофобия", "кацапы", "москали", "ватник", "вата",
                    "рашисты", "колорады", "русня", "русяки", "руслики",
                    "лапти", "лаптеногие", "мокшане",
                    "руссиш швайн", "рашкованы", "орки", "ордынцы", "чушки",
                    "ваньки", "бурятосы", "свиньи", "свинорылые",
                    "асвабадители", "окупанты", "тувинский олень",
                    "кастрюлеголовые", "лугандоны", "дамбас", "буча",
                    "на болоте"],
    "act_against_russia": ["санкции против России",
                           "бойкот российских товаров",
                           "противодействие России", "антироссийские меры",
                           "подрыв российских интересов", "ослабление России",
                           "изоляция России", "дамбить бамбас", "кремляди",
                           "путиноиды", "запутинцы", "зэтданутые", "зиганутые",
                           "рашкованы",
        "клятые москали", "жидовско-москальская хунта", "путинские ублюдки", "россия - террористическое государство",
        "кремлевские шавки", "путинская шайка", "российские оккупанты", "москальская орда", "путинские прихвостни",
        "кровавый режим", "российские фашисты", "путинские пособники", "москальское недоразумение", "путинские мартышки"
        ],
    "migrants": [
            "напали", "побили", "избили","подожгли"
        ],
    "regional_problems": [
        "взрыв", "хлопок", "поджог",
        "релейный шкаф", "диверсия",
        "задержали", "пострадавшие",
        "госпитализированы", "эвакуация"
    ],
    "others": [
           "пострадавших",
           "госпитализировали",
           "телефонные мошенники",
           "телефонные террористы",
           "призывы к экстремизму",
           "за призывы к экстремизму",
           "приговор",
           "экстремизм",
           "украинский агент",
           "организация экстремистского сообщества"
        ]
}


def encode_categories():
    encoded_categories = {}
    for model_name, model in models.items():
        encoded_categories[model_name] = {cat: model.encode(CATEGORIES[cat])
                                          for cat in CATEGORIES}
    return encoded_categories


encoded_categories = encode_categories()


def is_potentially_dangerous(text: str) -> bool:
    dangerous_keywords = ["угроза", "насилие", "незаконно", "преступление"]
    return any(keyword in text.lower() for keyword in dangerous_keywords)


def check_sensitive_topics(text: str, model) -> List[str]:
    embeddings = model.encode([text, *CATEGORIES.values()])
    text_embedding = embeddings[0]
    category_embeddings = embeddings[1:]
    similarities = util.pytorch_cos_sim(text_embedding, category_embeddings)[0]
    threshold = 0.5
    sensitive_topics = [list(CATEGORIES.keys())[i] for i, sim in
                        enumerate(similarities) if sim > threshold]
    return sensitive_topics


def is_extremist(text: Optional[str], max_similarity: float, model) -> Tuple[
    bool, Optional[str], Optional[str], float]:
    if text is None or text.strip() == "":
        return False, None, None, 0.0

    lower_text = text.lower()
    words = lower_text.split()

    # Fuzzy keyword matching
    for category, keywords in extremist_keywords.items():
        for word in words:
            for keyword in keywords:
                similarity = fuzz.ratio(word, keyword)
                if similarity > 60:  # Adjust this threshold as needed
                    return True, CATEGORIES[
                        category], keyword, similarity / 100.0

    # Semantic similarity check
    max_similarity_score = 0
    max_category = None

    text_embedding = model.encode(text)
    for category, category_embedding in encoded_categories[
        selected_model].items():
        similarity = util.pytorch_cos_sim(text_embedding,
                                          category_embedding).item()
        if similarity > max_similarity_score:
            max_similarity_score = similarity
            max_category = category

    # Check if the content is potentially dangerous
    if is_potentially_dangerous(text):
        return True, "Потенциально опасное содержание", None, max(
            max_similarity_score, 0.5)

    # Consider as potentially extremist if similarity score > max_similarity threshold
    if max_similarity_score > max_similarity:
        return True, CATEGORIES[max_category], None, max_similarity_score

    # Check sensitive topics
    sensitive_topics = check_sensitive_topics(text, model)
    if sensitive_topics:
        return True, f"Обнаружены чувствительные темы: {', '.join(sensitive_topics)}", None, 1.0

    return False, None, None, max_similarity_score


def process_message(message: Optional[str], max_similarity: float, model) -> \
Tuple[str, bool, Optional[str], Optional[str], float]:
    if message is None or message.strip() == "":
        return "", False, None, None, 0.0

    is_extremist_flag, category, matching_keyword, similarity_score = is_extremist(
        message, max_similarity, model)
    return message, is_extremist_flag, category, matching_keyword, similarity_score


# Основное приложение Streamlit
custom_message = st.text_area(
    "Введите пользовательское сообщение для проверки качества модели:")

if st.button("Анализировать"):
    if custom_message:
        model = list(models.values())[0]  # Get the selected model
        processed_message, is_extremist_flag, category, matching_keyword, similarity_score = process_message(
            custom_message, max_similarity, model)

        st.write(f"**Пользовательское сообщение:** {custom_message}")
        st.write(
            f"**Потенциально экстремистское:** {'Да' if is_extremist_flag else 'Нет'}")
        st.write(f"**Категория:** {category}")
        st.write(f"**Совпадающее ключевое слово:** {matching_keyword}")
        st.write(f"**Оценка сходства:** {similarity_score:.4f}")

        st.markdown("---")
    else:
        st.warning("Пожалуйста, введите сообщение для анализа.")


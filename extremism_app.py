
import streamlit as st
import re
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import numpy as np
from all_geo import GeoLocationNormalizer
import spacy
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel


# Load Spacy model
nlp = spacy.load("ru_core_news_md")

# Initialize the tokenizer and sentiment model
tokenizer = RegexTokenizer()
sentiment_model = FastTextSocialNetworkModel(tokenizer=tokenizer)

# Define the sentiment categories in the code
required_categories: List[str] = ['positive', 'negative', 'neutral', 'skip']

# Настройка страницы Streamlit
st.set_page_config(page_title="Обнаружение экстремистского контента",
                   layout="wide")
st.title("Обнаружение экстремистского контента")

# Боковая панель для выбора модели и параметров
st.sidebar.header("Выбор модели и параметров")
selected_model = st.sidebar.selectbox(
    "Выберите модель для анализа",
    ["rubert-base-cased-sentence"],
    index=0  # Установка 'distiluse' по умолчанию
)
# max_similarity = st.sidebar.slider("Порог максимального сходства", 0.0, 1.0,
#                                    0.3, 0.01)
max_similarity = 0.3


@st.cache(allow_output_mutation=True)
def load_models(model_choice):
    models = {}
    if model_choice == "rubert-base-cased-sentence":
        models['rubert-base-cased-sentence'] = SentenceTransformer(
            'DeepPavlov/rubert-base-cased-sentence')
    return models


models = load_models(selected_model)

normalizer = GeoLocationNormalizer('city_codes_cis.csv', 'city_numbers.csv',
                                   'world_city_codes.csv')
result = normalizer.get_normalized_list()


excluded = ["Ахнашин","хуайлай", "Камден", "Байша","Нашуа"]

result = [r for r in result if r not in excluded]
result.extend(["Крым", "Курск","крым", "курск","россия", "Pоссия"])
# List of Russian regions including Chechnya and Dagestan
russian_regions = [
    "Чечня",  # Chechnya
    "Дагестан",  # Dagestan
    "Москва",  # Moscow
    "Санкт-Петербург",  # Saint Petersburg
    "Татарстан",  # Tatarstan
    "Краснодарский край",  # Krasnodar Krai
    "Свердловская область",  # Sverdlovsk Oblast
    "Новосибирская область",  # Novosibirsk Oblast
    "Башкортостан",  # Bashkortostan
    "Ростовская область"  # Rostov Oblast
    # Add more regions as needed
]
result.extend(russian_regions)


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
    "nationalism": "Пропагандирует национализм или превосходство определенной национальности",
    "conspiracy_theories": "Распространяет теории заговора или призывает к их распространению",
    # "geolocations": "геопозиция",
    "others": "Другие",
    "vulgar": "Оскорбления и маты"
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
               "смерть русне","дно"],
    "minority_aggression": ["дискриминация меньшинств", "угнетение меньшинств",
                            "ненависть к мигрантам", "этническая чистка",
                            "расовая дискриминация", "ксенофобия"],
    "religious_extremism": ["религиозный фанатизм", "джихад",
                            "крестовый поход", "религиозная нетерпимость",
                            "сектантство", "религиозное насилие",
                            "богохульство",
                            "Аллах Акбар", "Аллах Велик", "Исламское государство", "Джихад", "Шариат", "Халифат",
                                "Мусульманские братья", "Талибан", "Хизб ут-Тахрир", "Аль-Каида", "Имарат Кавказ",
                                "Исламский джихад", "Боко Харам", "Джебхат ан-Нусра", "Ахрар аш-Шам", "Ансар аль-Шариа",
                                "Ахрар аль-Шам", "Джамаат Ансар Дин", "Джамаат Таухид валь-Джихад", "Исламская партия Туркестана",
                                "Исламское движение Узбекистана", "Исламское движение Восточного Туркестана", "Исламское движение Уйгуров",
                                "Движение талибов", "Талибан", "Исламский эмират Афганистан", "Исламская государственность",
                                "Исламское государство Ирака и Леванта", "Исламское государство в Ираке и Леванте",
                                "Исламское государство Ирака", "Исламское государство Леванта", "Исламское государство", "ИГИЛ",
                                "ИГ", "ДАИШ", "ИГИШ", "ИГИЛ", "Исламское государство в Ливии", "Исламское государство в Египте"
                            ],
    "russophobia": ["русофобия", "кацапы", "москали", "ватник", "вата",
                    "рашисты", "колорады", "русня", "русяки", "руслики",
                    "лапти", "лаптеногие", "мокшане",
                    "руссиш швайн", "рашкованы", "орки", "ордынцы", "чушки",
                    "ваньки", "бурятосы", "свиньи", "свинорылые",
                    "асвабадители", "окупанты", "тувинский олень",
                    "кастрюлеголовые", "лугандоны", "дамбас", "буча",
                    "на болоте", "еблан", "рабы","хуйло"],
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
    "nationalism": [
        "русские выше всех", "национализм", "национальное превосходство",
        "великая нация", "чистота нации", "национальная исключительность",
        "этническое превосходство", "расовая чистота", "этническая гордость",
        "национальная гордость", "национальное возрождение", "этническое самосознание",
        "национальное самосознание", "национальная идентичность", "этническая идентичность",
        "национальное единство", "этническое единство", "национальное государство",
        "этническое государство", "национальная солидарность", "этническая солидарность"
    ],

    "conspiracy_theories": [
        "за всем стоят евреи","иуде́о-масо́нский заговор", "Жидомасо́нский заговор",
        "теория заговора", "заговор", "закулисные силы", "тайные силы", "мировой заговор",
        "новый мировой порядок", "Нью-йорк", "11 сентября", "Пентагон", "Черное солнце",
        "сионисты", "протоколы сионских мудрецов", "иллюминаты", "масоны", "рептилоиды",
        "переворот", "государственный переворот", "революция", "мировая закулиса", " Globe of Treason",
        "Великий Архитектор Всего", "Золотой миллиард", "Комитет 300", "Совет по международным отношениям",
        "Нью-Йоркская Комиссия по глобальным стратегиям", "Бильдерберг", "Давос", "Веллингтонский клуб",
        "Клуб Ротшильд", "Клуб Римских Клубов", "Бордерленд", "Пиктевский замок", "Чатеaux de Chantilly",
        "Берклийские институты", "Трёхсторонняя комиссия", "Совет национальной безопасности",
        "Совет по национальной политике", "ФРС", "ФБР", "СВР", "СБУ", "ГРУ", "ПАП", "ПЦР", "РНЕФ", "ФСБ", "ФСО",
        "Минобороны", "ГОС", "СВР", "СБУ", "ГРУ", "ПАП", "ПЦР", "РНЕФ", "ФСБ", "ФСО", "Минобороны", "ГОС",
        "СВР", "СБУ", "ГРУ", "ПАП", "ПЦР", "РНЕФ", "ФСБ", "ФСО", "Минобороны", "ГОС"
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
           "организация экстремистского сообщества",
            "враг",
            "гавно"

        ],

    # "geolocations": result,
    "vulgar":
        ["хуй","хуйло","чмо"]


}

def analyze_sentiment(text: Optional[str]) -> Dict[str, float]:
    if not text:
        return {category: 0.0 for category in required_categories}
    results = sentiment_model.predict([text], k=2)[0]
    for category in required_categories:
        results.setdefault(category, 0.0)
    sorted_results = {category: results[category] for category in required_categories}
    return sorted_results

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

    # Add more neutral or positive phrases here
    neutral_phrases = {
        "крым россии": "Нейтральное или положительное содержание",
        "курск это россия": "Нейтральное содержание",
        # Add more phrases as needed
    }

    # Check for neutral or positive phrases
    for phrase, category in neutral_phrases.items():
        if phrase in lower_text:
            return False, category, phrase, 0.0

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


def is_relevant_for_extremism(message: str,
                              extremist_keywords: Dict[str, List[str]],
                              model) -> bool:
    lower_message = message.lower()

    positive_words = ["молодец"]
    if lower_message in positive_words or message in positive_words:
        return False

    # Проверка на точное совпадение ключевых слов
    for category_keywords in extremist_keywords.values():
        if any(keyword.lower() in lower_message for keyword in
               category_keywords):
            return True

    # Проверка на нечеткое совпадение (fuzz.ratio > 0.6)
    words = lower_message.split()
    for category_keywords in extremist_keywords.values():
        for word in words:
            for keyword in category_keywords:
                if fuzz.ratio(word, keyword.lower()) > 90:
                    return True

    # Проверка на семантическое сходство
    message_embedding = model.encode(message)
    for category, keywords in extremist_keywords.items():
        category_embedding = model.encode(CATEGORIES[category])
        similarity = util.pytorch_cos_sim(message_embedding,
                                          category_embedding).item()
        if similarity > 0.9:  # Можно настроить этот порог
            return True

    sentiment_result = analyze_sentiment(lower_message)
    sentiment_category = max(sentiment_result, key=sentiment_result.get)

    if sentiment_category in ['positive']:
        return False




    return False

def process_message(message: Optional[str], max_similarity: float, model) -> Tuple[str, bool, Optional[str], Optional[str], float, bool]:
    if message is None or message.strip() == "":
        return "", False, None, None, 0.0, False

    is_relevant = is_relevant_for_extremism(message, extremist_keywords, model)
    if not is_relevant:
        return message, False, None, None, 0.0, False

    is_extremist_flag, category, matching_keyword, similarity_score = is_extremist(message, max_similarity, model)
    return message, is_extremist_flag, category, matching_keyword, similarity_score, True
# Основное приложение Streamlit
custom_message = st.text_area(
    "Введите пользовательское сообщение для проверки качества модели:")

if st.button("Анализировать"):
    if custom_message:
        model = list(models.values())[0]  # Получаем выбранную модель
        processed_message, is_extremist_flag, category, matching_keyword, similarity_score, is_relevant = process_message(custom_message, max_similarity, model)
        sentiment_result = analyze_sentiment(custom_message)
        sentiment_category = max(sentiment_result, key=sentiment_result.get)
        if custom_message in ["Путин", "nутин"]:
            sentiment_category='neutral'


        print(custom_message, sentiment_category)



        if not is_relevant and is_extremist_flag is False:
            st.warning("Сообщение не релевантно для обнаружения экстремизма.")
        elif sentiment_category=='positive':
            st.warning("Сообщение не релевантно для обнаружения экстремизма.")
        elif sentiment_category=='neutral':
            st.warning("Сообщение не релевантно для обнаружения экстремизма.")

        else:
            st.write(f"**Пользовательское сообщение:** {custom_message}")
            st.write(f"**Потенциально экстремистское:** {'Да' if is_extremist_flag else 'Нет'}")
            st.write(f"**Категория:** {category}")
            st.write(f"**Совпадающее ключевое слово:** {matching_keyword}")
            st.write(f"**Оценка сходства:** {similarity_score:.4f}")

        st.markdown("---")
    else:
        st.warning("Пожалуйста, введите сообщение для анализа.")

from typing import List
from category_taxonomy import CATEGORY_TAXONOMY

def translate_categories(categories: List[str]) -> List[str]:
    """
    Given a list of category short codes, return a list of corresponding full category names.

    :param categories: List of category short codes (e.g. 'cs.AI', 'stat.ML').
    :return: List of category full names (e.g. 'Artificial Intelligence', 'Machine Learning').
    """
    translated_categories = []

    for category in categories:
        translated = CATEGORY_TAXONOMY.get(category, category)  # If not found, keep the original
        translated_categories.append(translated)

    return translated_categories

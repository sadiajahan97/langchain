import requests
from typing import Any, Callable, TypedDict

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    dynamic_prompt,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from langchain.messages import SystemMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from wikipediaapi import Wikipedia


class Context(TypedDict):
    is_premium: bool
    user_role: str


class CountryCapitalInfo(TypedDict, total=False):
    latlng: list[float]


class CountryCar(TypedDict, total=False):
    side: str
    signs: list[str]


class CountryCoatOfArms(TypedDict, total=False):
    png: str
    svg: str


class CountryCurrency(TypedDict, total=False):
    name: str
    symbol: str


class CountryDemonym(TypedDict, total=False):
    f: str
    m: str


class CountryFlags(TypedDict, total=False):
    alt: str
    png: str
    svg: str


class CountryIdd(TypedDict, total=False):
    root: str
    suffixes: list[str]


class CountryName(TypedDict, total=False):
    common: str
    nativeName: dict[str, "CountryNameNative"]
    official: str


class CountryNameNative(TypedDict, total=False):
    common: str
    official: str


class CountryPostalCode(TypedDict, total=False):
    format: str
    regex: str


class CountryResponse(TypedDict, total=False):
    altSpellings: list[str]
    area: float
    borders: list[str]
    capital: list[str]
    capitalInfo: CountryCapitalInfo
    car: CountryCar
    cca2: str
    cca3: str
    ccn3: str
    cioc: str
    coatOfArms: CountryCoatOfArms
    continents: list[str]
    currencies: dict[str, CountryCurrency]
    demonyms: dict[str, CountryDemonym]
    fifa: str
    flag: str
    flags: CountryFlags
    gini: dict[str, float]
    idd: CountryIdd
    independent: bool
    landlocked: bool
    languages: dict[str, str]
    latlng: list[float]
    maps: dict[str, str]
    name: CountryName
    population: int
    postalCode: CountryPostalCode
    region: str
    startOfWeek: str
    status: str
    subregion: str
    timezones: list[str]
    tld: list[str]
    translations: dict[str, "CountryNameNative"]
    unMember: bool


class CustomMiddleware(AgentMiddleware):
    def before_model(self, state: "CustomState", runtime) -> dict[str, Any] | None:
        pass

    state_schema = "CustomState"


class CustomState(AgentState):
    user_preferences: dict


advanced_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

basic_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

system_prompt = SystemMessage(
    content=[
        {
            "type": "text",
            "text": "You are a helpful geography assistant. Use the provided tools to fetch comprehensive and factual information about countries when users ask for it.",
        },
    ]
)


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.get("user_role", "user")

    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} You are an expert in your field. Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} You are a beginner. Explain concepts simply and avoid jargon."

    return base_prompt


@tool
def get_country_by_name(country_name: str) -> list[CountryResponse]:
    """
    Search for a country by its name using the RestCountries API.

    Args:
        country_name (str): The name of the country to search for.

    Returns:
        list[CountryResponse]: A list of dictionaries representing the Country objects returned by the API.

    Response Body Format / Fields Info:
    - alpha2Code / cca2: ISO 3166-1 alpha-2 two-letter country codes
    - alpha3Code / cca3: ISO 3166-1 alpha-3 three-letter country codes
    - altSpellings: Alternate spellings of the country name
    - area: Geographical size
    - borders: Border countries
    - callingCodes / idd: International dialing codes
    - capital: Capital cities
    - capitalInfo > latlng: Capital latitude and longitude
    - car > side: Car driving side
    - car > signs: Car distinguised (oval) signs
    - cioc: Code of the International Olympic Committee
    - coatOfArms: MainFacts.com links to svg and png images
    - continents: List of continents the country is on
    - currencies: List of all currencies
    - demonyms (m/f): Genderized inhabitants of the country
    - fifa: FIFA code
    - flag: Link to the svg flag on Flagpedia or flag emoji
    - flags: Flagpedia links to svg and png flags
    - gini: Worldbank Gini index
    - independent: ISO 3166-1 independence status (the country is considered a sovereign state)
    - landlocked: Landlocked country
    - languages: List of official languages
    - latlng: Latitude and longitude
    - maps: Link to Google maps and Open Street maps
    - name: Country name
    - name > nativeName > official/common: Official and common native country name
    - name > official/common: Official and common country name
    - numericCode / ccn3: ISO 3166-1 numeric code (UN M49)
    - population: Country population
    - postalCode > format/regex: Country postal codes
    - region: UN demographic regions
    - startOfWeek: Day of the start of week (Sunday/Monday/Saturday)
    - status: ISO 3166-1 assignment status
    - subregion: UN demographic subregions
    - timezones: Timezones
    - topLevelDomain / tld: Internet top level domains
    - translations: List of country name translations
    - unMember: UN Member status
    """
    response = requests.get(f"https://restcountries.com/v3.1/name/{country_name}")

    response.raise_for_status()

    return response.json()


@tool
def get_data_from_wikipedia(language: str, query: str):
    """
    Retrieves the summary of a Wikipedia page based on a search query.

    This function searches Wikipedia in the specified language for the given query,
    retrieves the top search result, and extracts the summary text of that page.
    It uses a custom user agent to comply with Wikipedia's request policies.

    Args:
        language (str): The language code for the Wikipedia edition to search
            (e.g., 'en' for English, 'bn' for Bengali, 'fr' for French).
        query (str): The search term or topic to look up on Wikipedia.

    Returns:
        str: The introductory summary text of the most relevant Wikipedia page found.

    Raises:
        IndexError: If no matching Wikipedia pages are found (the results list is empty).
        Exception: Built-in library exceptions if there are issues connecting
            to the Wikipedia API or retrieving the page content.
    """
    wikipedia = Wikipedia(
        language=language,
        user_agent="WikipediaTool/1.0 (https://github.com/sadiajahan97/langchain; sadiaiffatjahan@gmail.com) python-requests/2.33.1",
    )

    results = wikipedia.search(limit=1, query=query)

    title = list(results.pages.keys())[0]

    page = wikipedia.page(title=title)

    return page.summary


@wrap_model_call
def dynamic_model_selection(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    is_premium = len(request.runtime.context.get("is_premium", False))

    model = advanced_model if is_premium else basic_model

    return handler(request.override(model=model))


agent = create_agent(
    context_schema=Context,
    # middleware=[dynamic_model_selection, user_role_prompt, CustomMiddleware()],
    model=basic_model,
    name="ai_agent",
    system_prompt=system_prompt,
    tools=[get_country_by_name],
)

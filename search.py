import requests

def web_search(query):
    try:
        url = f"https://duckduckgo.com/html/?q={query}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            text = r.text
            start = text.find("<a rel=\"nofollow\"")
            if start != -1:
                snippet = text[start:start+200]
                return snippet.replace("\n", " ")[:200]
        return "Não encontrei resultados."
    except Exception as e:
        return f"Erro ao buscar: {e}"

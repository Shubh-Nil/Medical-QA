
from __future__ import annotations
import os, time, typing, requests, xml.etree.ElementTree as ET
from dataclasses import dataclass

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EMAIL = os.environ.get("PUBMED_EMAIL", "")
API_KEY = os.environ.get("PUBMED_API_KEY", "")
USER_AGENT = "MedQA-RAG-QLoRA/0.1 (+https://example.org)"

@dataclass
class PubMedRecord:
    pmid: str
    title: str
    abstract: str
    journal: str | None = None
    year: str | None = None
    doi: str | None = None
    url: str | None = None

class PubMedClient:
    def __init__(self, pause: float = 0.34):
        self.pause = pause  # be polite (<= 3 req/s without key)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _params(self, extra: dict) -> dict:
        base = {"email": EMAIL}
        if API_KEY:
            base["api_key"] = API_KEY
        base.update(extra)
        return base

    def search(self, query: str, retmax: int = 20) -> list[str]:
        """Return a list of PMIDs for a query."""
        time.sleep(self.pause)
        r = self.session.get(f"{EUTILS}/esearch.fcgi", params=self._params({
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
        }), timeout=30)
        r.raise_for_status()
        js = r.json()
        return js.get("esearchresult", {}).get("idlist", [])

    def fetch(self, pmids: list[str]) -> list[PubMedRecord]:
        """Fetch records by PMID (xml -> dataclass)."""
        if not pmids:
            return []
        time.sleep(self.pause)
        r = self.session.get(f"{EUTILS}/efetch.fcgi", params=self._params({
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }), timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        out: list[PubMedRecord] = []
        for art in root.findall(".//PubmedArticle"):
            pmid = (art.findtext(".//PMID") or "").strip()
            title = (art.findtext(".//ArticleTitle") or "").strip()
            abstract = " ".join([t.text or "" for t in art.findall(".//Abstract/AbstractText")]).strip()
            journal = (art.findtext(".//Journal/Title") or "").strip() or None
            year = (art.findtext(".//PubDate/Year") or "").strip() or None
            doi = None
            for idn in art.findall(".//ArticleIdList/ArticleId"):
                if (idn.attrib or {}).get("IdType","").lower() == "doi":
                    doi = (idn.text or "").strip()
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
            out.append(PubMedRecord(pmid=pmid, title=title, abstract=abstract, journal=journal, year=year, doi=doi, url=url))
        return out

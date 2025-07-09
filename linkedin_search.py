import os, json, requests, re
from bs4 import BeautifulSoup

GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID")

def fetch_company_people(company: str, size: int = 10):
    q = f"site:linkedin.com/in \"{company}\""
    resp = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"key": GOOGLE_CSE_KEY, "cx": GOOGLE_CSE_ID, "q": q, "num": size},
        timeout=5,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    out = []

    for item in items:
        name = item.get("title", "").replace(" | LinkedIn", "").strip()
        url  = item.get("link")

        # 1) Try extracting headshot & current role from JSON-LD
        image = None
        role  = None
        try:
            profile = requests.get(url, timeout=5)
            profile.raise_for_status()
            soup = BeautifulSoup(profile.text, "html.parser")
            ld = soup.find("script", {"type":"application/ld+json"})
            if ld:
                data = json.loads(ld.string)
                # Grab the headshot if available
                image = data.get("image") or image
                # Grab the jobTitle if they’re at our company
                if data.get("worksFor",{}).get("name","").lower() == company.lower():
                    role = data.get("jobTitle")
        except Exception:
            # any network/parsing error → fall back below
            pass

        # 2) Fallback: pull “Senior Dev at Infosys” out of the snippet
        if not role:
            raw_snip = item.get("snippet","").strip()
            pattern = rf"(.+? at {re.escape(company)})"
            m = re.search(pattern, raw_snip, flags=re.IGNORECASE)
            role = m.group(1).strip() if m else raw_snip.split(".")[0].strip()

        # 3) Fallback for image: maybe CSE pagemap thumbnail?
        #    (Google sometimes returns `pagemap` → `cse_thumbnail` or `cse_image`.)
        if not image:
            pagemap = item.get("pagemap",{})
            for fld in ("cse_thumbnail","cse_image"):
                if fld in pagemap and len(pagemap[fld])>0 and pagemap[fld][0].get("src"):
                    image = pagemap[fld][0]["src"]
                    break

        out.append({
            "name":  name,
            "url":   url,
            "image": image,
            "role":  role
        })

    return out

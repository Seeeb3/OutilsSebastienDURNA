import asyncio
from playwright.async_api import async_playwright
import pandas as pd

async def scrape_with_playwright():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=100)
        page = await browser.new_page()
        await page.goto("https://www.tdcj.texas.gov/death_row/dr_executed_offenders.html")
        await page.wait_for_selector("table.tdcj_table", timeout=60000)

        rows = await page.evaluate('''() => {
            const table = document.querySelector("table.tdcj_table");
            const data = [];
            for (const row of Array.from(table.querySelectorAll("tr")).slice(1)) {
                const cells = row.querySelectorAll("td");
                if (cells.length < 10) continue;
                data.push({
                    execution_id: parseInt(cells[0].innerText.trim()),
                    inmate_info_url: cells[1].querySelector("a")?.href || "",
                    last_statement_url: cells[2].querySelector("a")?.href || "",
                    last_name: cells[3].innerText.trim(),
                    first_name: cells[4].innerText.trim(),
                    tdcj_number: cells[5].innerText.trim(),
                    age: parseInt(cells[6].innerText.trim()),
                    date: cells[7].innerText.trim(),
                    race: cells[8].innerText.trim(),
                    county: cells[9].innerText.trim(),
                });
            }
            return data;
        }''')

        for row in rows:
            info_page = await browser.new_page()
            try:
                if row["inmate_info_url"].endswith(".html"):
                    await info_page.goto(row["inmate_info_url"])
                    row["info_details"] = await info_page.evaluate('''() => {
                        const table = document.querySelector("#content_right > table");
                        if (!table) return {};
                        const data = {};
                        for (const tr of table.querySelectorAll("tr")) {
                            const tds = tr.querySelectorAll("td");
                            if (tds.length === 2) {
                                const key = tds[0].innerText.trim();
                                const value = tds[1].innerText.trim();
                                data[key] = value;
                            }
                        }
                        return data;
                    }''')
                else:
                    row["info_details"] = {}
            except:
                row["info_details"] = {}
            await info_page.close()

            statement_page = await browser.new_page()
            try:
                if row["last_statement_url"].endswith(".html"):
                    await statement_page.goto(row["last_statement_url"])
                    row["last_statement"] = await statement_page.evaluate('''() => {
                        const el = document.querySelector("#content_right > p:nth-child(11)");
                        return el ? el.innerText.trim() : "";
                    }''')
                else:
                    row["last_statement"] = ""
            except:
                row["last_statement"] = ""
            await statement_page.close()

        for row in rows:
            info_data = row.pop("info_details", {})
            for k, v in info_data.items():
                row[k] = v

        print(f"Terminé. {len(rows)} condamnés traités.")
        await browser.close()
        df = pd.DataFrame(rows)
        df.to_csv("data/raw/death_row_index.csv", index=False)
        print("Fichier death_row_index.csv généré")

if __name__ == "__main__":
    asyncio.run(scrape_with_playwright())
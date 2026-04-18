from playwright.sync_api import sync_playwright
import time
import os

def take_screenshots():
    os.makedirs("assets", exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Use a large viewport to get a nice screenshot
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        
        print("Navigating to http://localhost:8502...")
        page.goto("http://localhost:8502", timeout=60000)
        
        # Wait for the main title
        page.wait_for_selector("text=Yield Curve & Portfolio Manager", timeout=30000)
        print("Page loaded.")
        time.sleep(5)  # Let any animations/charts render
        
        print("Taking default curve screenshot...")
        page.screenshot(path="assets/app_curves.png", full_page=True)
        
        print("Switching to Portfolio tab...")
        page.click('button[role="tab"]:has-text("Portfolio")')
        time.sleep(3)
        page.screenshot(path="assets/app_portfolio.png", full_page=True)
        
        print("Switching to Carry & Roll-Down tab...")
        page.click('button[role="tab"]:has-text("Carry & Roll-Down")')
        time.sleep(3)
        page.screenshot(path="assets/app_carry_rolldown.png", full_page=True)
        
        print("Switching to relative value tab...")
        page.click('button[role="tab"]:has-text("Relative Value")')
        time.sleep(3)
        page.screenshot(path="assets/app_relative_value.png", full_page=True)
        
        print("Switching to P&L...")
        page.click('button[role="tab"]:has-text("P&L Attribution")')
        time.sleep(3)
        page.screenshot(path="assets/app_pnl.png", full_page=True)
        
        browser.close()
        print("Screenshots taken successfully.")

if __name__ == "__main__":
    take_screenshots()

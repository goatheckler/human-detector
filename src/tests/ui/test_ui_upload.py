import pytest
import time
from pathlib import Path
from playwright.sync_api import Page, expect

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "images"
UI_URL = "http://localhost:8501"

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
    }

def wait_for_streamlit(page: Page):
    page.goto(UI_URL)
    page.wait_for_selector("text=Human Detector", timeout=10000)

def upload_file_and_detect(page: Page, file_path: Path):
    # Upload file (CPU is already the default device)
    file_input = page.locator('input[type="file"]')
    file_input.set_input_files(str(file_path))
    
    # Give Streamlit time to process the file upload
    time.sleep(2)
    
    detect_button = page.get_by_role("button", name="🔍 Detect Humans")
    expect(detect_button).to_be_visible()
    
    # Click and wait for Streamlit to process
    detect_button.click()
    
    # Check if an error occurred
    try:
        error = page.locator('[data-testid="stException"]')
        if error.is_visible(timeout=2000):
            raise Exception(f"Streamlit error: {error.text_content()}")
    except:
        pass  # No error, continue
    
    # Wait for results with extended timeout
    page.wait_for_selector("text=Detection Result", timeout=30000)

def test_ui_loads(page: Page):
    wait_for_streamlit(page)
    expect(page.locator("text=Human Detector")).to_be_visible()
    expect(page.locator("text=Upload Your Image")).to_be_visible()

def test_upload_image_without_humans_gradient(page: Page):
    wait_for_streamlit(page)
    
    image_path = FIXTURES_DIR / "without_humans" / "gradient.jpg"
    upload_file_and_detect(page, image_path)
    
    expect(page.locator("text=❌ No Human Detected")).to_be_visible()
    expect(page.locator("text=0.00%")).to_be_visible()

def test_upload_stick_figure_not_detected(page: Page):
    wait_for_streamlit(page)
    
    image_path = FIXTURES_DIR / "with_humans" / "stick_figure.jpg"
    upload_file_and_detect(page, image_path)
    
    expect(page.locator("text=❌ No Human Detected")).to_be_visible()

def test_timing_metric_displayed(page: Page):
    wait_for_streamlit(page)
    
    image_path = FIXTURES_DIR / "without_humans" / "gradient.jpg"
    upload_file_and_detect(page, image_path)
    
    expect(page.locator("text=Analysis Time")).to_be_visible()
    expect(page.locator("text=/\\d+ms/")).to_be_visible()

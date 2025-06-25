import pytest
import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch
from browser.context import BrowserContext, BrowserContextConfig, BrowserSession
from browser.views import URLNotAllowedError, BrowserError
from dom.views import DOMElementNode


@pytest.fixture
def browser_mock():
    browser = MagicMock()
    browser.config = MagicMock()
    browser.config.cdp_url = None
    browser.config.chrome_instance_path = None
    return browser


@pytest.fixture
def context_config():
    return BrowserContextConfig(
        cookies_file=None, allowed_domains=["google.com", "ruc.edu.cn", "baidu.com"]
    )


@pytest.fixture
def browser_context(browser_mock, context_config):
    return BrowserContext(browser_mock, config=context_config)


@pytest.mark.asyncio
async def test_is_url_allowed(browser_context):
    # Test allowed domains
    assert browser_context._is_url_allowed("https://google.com")
    assert browser_context._is_url_allowed("https://scholar.google.com")


@pytest.mark.asyncio
async def test_navigate_to_allowed_url(browser_context: BrowserContext):
    page_mock = AsyncMock()
    browser_context.get_current_page = AsyncMock(return_value=page_mock)

    await browser_context.navigate_to("https://www.ruc.edu.cn")

    page_mock.goto.assert_called_once_with("https://www.ruc.edu.cn")
    page_mock.wait_for_load_state.assert_called_once()


@pytest.mark.asyncio
async def test_navigate_to_disallowed_url(browser_context):
    with pytest.raises(BrowserError):
        await browser_context.navigate_to("https://malicious.com")


@pytest.mark.asyncio
async def test_switch_to_tab(browser_context):
    # Setup mocks
    page_mock = AsyncMock()
    page_mock.url = "http://www.ai.ruc.edu.cn"
    context_mock = MagicMock()
    context_mock.pages = [page_mock]
    session_mock = MagicMock()
    session_mock.context = context_mock

    browser_context.get_session = AsyncMock(return_value=session_mock)

    # Test valid tab switch
    await browser_context.switch_to_tab(0)

    page_mock.bring_to_front.assert_called_once()
    page_mock.wait_for_load_state.assert_called_once()
    assert session_mock.current_page == page_mock


@pytest.mark.asyncio
async def test_switch_to_invalid_tab(browser_context):
    session_mock = MagicMock()
    session_mock.context = MagicMock()
    session_mock.context.pages = []
    browser_context.get_session = AsyncMock(return_value=session_mock)

    with pytest.raises(BrowserError):
        await browser_context.switch_to_tab(0)


@pytest.mark.asyncio
async def test_create_new_tab(browser_context: BrowserContext):
    # Setup mocks
    new_page_mock = AsyncMock()
    context_mock = AsyncMock()
    context_mock.new_page.return_value = new_page_mock
    session_mock = MagicMock()
    session_mock.context = context_mock

    browser_context.get_session = AsyncMock(return_value=session_mock)
    browser_context._wait_for_page_and_frames_load = AsyncMock()

    # Test creating new tab with URL
    await browser_context.create_new_tab(
        "https://gsai.ruc.edu.cn/addons/teacher/index.html"
    )

    context_mock.new_page.assert_called_once()
    new_page_mock.wait_for_load_state.assert_called_once()
    new_page_mock.goto.assert_called_once_with(
        "https://gsai.ruc.edu.cn/addons/teacher/index.html"
    )


@pytest.mark.asyncio
async def test_get_tabs_info(browser_context: BrowserContext):
    # Setup mocks
    page1 = AsyncMock()
    page1.url = "https://gsai.ruc.edu.cn/addons/teacher/index.html"
    page1.title.return_value = "Example"

    page2 = AsyncMock()
    page2.url = "https://www.baidu.com"
    page2.title.return_value = "Test"

    context_mock = MagicMock()
    context_mock.pages = [page1, page2]
    session_mock = MagicMock()
    session_mock.context = context_mock

    browser_context.get_session = AsyncMock(return_value=session_mock)

    tabs_info = await browser_context.get_tabs_info()
    with open("tabs_info.txt", "w") as f:
        f.write(str(tabs_info))

    assert len(tabs_info) == 2
    assert tabs_info[0].url == "https://gsai.ruc.edu.cn/addons/teacher/index.html"
    assert tabs_info[0].page_id == 0
    assert tabs_info[1].url == "https://www.baidu.com"
    assert tabs_info[1].page_id == 1


@pytest.mark.asyncio
async def test_enhanced_css_selector_generation():
    # Test basic element
    element = DOMElementNode(
        tag_name="div",
        xpath="/html/body/div[1]",
        attributes={"class": "test-class", "id": "test-id"},
        is_visible=True,
        parent=None,
        children=[],
    )

    selector = BrowserContext._enhanced_css_selector_for_element(element)
    assert "div" in selector
    assert ".test-class" in selector
    assert '[id="test-id"]' in selector

    # Test element with special characters
    element = DOMElementNode(
        tag_name="input",
        xpath="/html/body/input[1]",
        attributes={"data-testid": "test:id"},
        is_visible=True,
        parent=None,
        children=[],
    )

    selector = BrowserContext._enhanced_css_selector_for_element(element)
    assert "input" in selector
    assert '[data-testid="test:id"]' in selector


@pytest.mark.asyncio
async def test_wait_for_stable_network(browser_context):
    page_mock = AsyncMock()
    browser_context.get_current_page = AsyncMock(return_value=page_mock)

    # Test normal network stabilization
    await browser_context._wait_for_stable_network()

    # Verify event listeners were added and removed
    assert page_mock.on.call_count == 2  # request and response
    assert page_mock.remove_listener.call_count == 2


if __name__ == "__main__":
    pytest.main(["-v"])

import os
import random
import time
import asyncio
from model import Node, Tree
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from playwright.async_api import (
    async_playwright,
    Playwright,
    Browser,
    Page,
    Frame,
    Dialog,
    BrowserContext,
)

from urllib.parse import urlparse, urljoin
import re
import sys
import glob

# (Keep USER_AGENTS definition if it's defined elsewhere, or define it here)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    # Add more user agents
]


async def random_sleep(
    min_duration=0.1, max_duration=0.5
):  # Slightly shorter sleeps might be okay
    await asyncio.sleep(random.uniform(min_duration, max_duration))


def handle_dialog(dialog: Dialog):
    """处理所有类型的对话框"""
    print(f"Dismissing dialog: {dialog.type} - {dialog.message}")
    dialog.dismiss()  # Or dialog.accept() if that's more appropriate for some dialogs


# JavaScript to be injected for removing popups
POPUP_REMOVER_SCRIPT = """
() => {
    const isLikelyNavbarOrHeader = (element) => {
        // ... (paste the full isLikelyNavbarOrHeader JS function here)
        if (!element || typeof element.getBoundingClientRect !== 'function') return false;
        const tagName = element.tagName.toLowerCase();
        const classNames = (element.className || '').toLowerCase();
        const id = (element.id || '').toLowerCase();
        // const textContent = (element.textContent || '').toLowerCase(); // Can be expensive
        const rect = element.getBoundingClientRect();

        if (rect.width === 0 || rect.height === 0) return false; // Skip invisible elements

        const isAtTop = rect.top < 100 && rect.height < 250; // Heuristic for top nav
        const isAtBottomSticky = rect.bottom > (window.innerHeight - 100) && rect.top > (window.innerHeight - 250);

        const navKeywords = ['nav', 'header', 'menu', 'navigation', 'toolbar', 'topbar', 'menubar', 'breadcrumb', 'top-menu', 'main-nav'];
        if (navKeywords.some(keyword => classNames.includes(keyword) || id.includes(keyword) || tagName === keyword)) {
            if (isAtTop || isAtBottomSticky || window.getComputedStyle(element).position === 'sticky' || window.getComputedStyle(element).position === 'fixed') {
                if (element.querySelectorAll('a, li').length > 2 || element.querySelector('ul, ol')) return true;
            }
        }
        if ((tagName === 'nav' || tagName === 'header' || tagName === 'menu') && (isAtTop || isAtBottomSticky)){
            return true;
        }
        if (window.getComputedStyle(element).position === 'fixed' && rect.width >= window.innerWidth * 0.8 && rect.height < 200 && isAtTop) {
             if (element.querySelectorAll('a, li').length > 1) return true;
        }
        if (id === 'wpadminbar') return true; // WordPress admin bar
        return false;
    };

    const isLikelyCookieBanner = (element) => {
        // ... (paste the full isLikelyCookieBanner JS function here)
        if (!element || typeof element.getBoundingClientRect !== 'function') return false;
        const text = (element.textContent || '').toLowerCase();
        const keywords = ['cookie', 'consent', 'privacy', 'accept', 'gdpr', 'ccpa', 'manage settings', 'preferences', 'data policy', 'we use cookies'];
        if (keywords.some(kw => text.includes(kw))) {
            const rect = element.getBoundingClientRect();
            const style = window.getComputedStyle(element);
            if (style.position === 'fixed' || style.position === 'sticky' || style.position === 'absolute') { // Added absolute
                if ( (rect.bottom > (window.innerHeight - 300) && rect.height < 400) || // Bottom quarter of screen, not too tall
                     (rect.top < 300 && rect.height < 400) || // Top quarter
                     (rect.width > window.innerWidth * 0.5 && rect.height < window.innerHeight * 0.6) ) { // Takes up decent screen space
                    return true;
                }
            }
            const zIndex = parseInt(style.zIndex, 10) || 0;
            if (zIndex > 100 && rect.width > window.innerWidth * 0.3 && rect.height > window.innerHeight * 0.1) {
                 return true;
            }
        }
        return false;
    };

    const elementsToRemove = new Set();
    let removedCount = 0;

    // Function to attempt removal, respects parentNode check
    const tryRemove = (el, reason) => {
        if (el && el.parentNode && !elementsToRemove.has(el)) {
            if (isLikelyNavbarOrHeader(el)) {
                // console.log('Skipping removal of likely navbar/header:', el);
                return;
            }
            // console.log('Marking for removal:', reason, el);
            elementsToRemove.add(el);
        }
    };

    // 1. Specific selectors for popups, modals, dialogs, overlays
    const popupSelectors = [
        '.modal', '.popup', '.dialog', '.overlay', '.lightbox', '.modal-backdrop', '.modal-dialog',
        '[class*="modal"]', '[class*="popup"]', '[class*="dialog"]', '[class*="overlay"]', '[class*="lightbox"]',
        '[id*="modal"]', '[id*="popup"]', '[id*="dialog"]', '[id*="overlay"]', '[id*="lightbox"]',
        '[role="dialog"]', '[aria-modal="true"]',
        // Cookie specific selectors
        '#onetrust-consent-sdk', '[id*="cookiebanner"]', '[class*="cookiebanner"]',
        '[id*="cookienotice"]', '[class*="cookienotice"]', '[class*="optanon-alert-box"]',
        // Common ad/blocker selectors (be careful not to be too aggressive)
        '[class*="ad"]', '[id*="ad"]', // Very generic, might remove legitimate content, use with caution or more specific checks
        '[aria-label*="close ad"]'
    ];
    popupSelectors.forEach(selector => {
        try {
            document.querySelectorAll(selector).forEach(el => tryRemove(el, `selector: ${selector}`));
        } catch (e) { /* console.error('Error with selector:', selector, e); */ }
    });

    // 2. Generic check for fixed/sticky elements that are likely popups or full-page overlays
    document.querySelectorAll('*').forEach(el => {
        try {
            const style = window.getComputedStyle(el);
            if (!style || !el.getBoundingClientRect) return; // Element might have been removed or is not visible

            const position = style.position;
            const zIndex = parseInt(style.zIndex, 10) || 0;
            const rect = el.getBoundingClientRect();

            if (rect.width === 0 || rect.height === 0 || style.display === 'none' || style.visibility === 'hidden') {
                return; // Skip invisible or zero-size elements
            }

            if (isLikelyCookieBanner(el)) {
                tryRemove(el, 'likely cookie banner');
                return; // Already handled
            }

            if (position === 'fixed' || position === 'sticky') {
                const isFullScreenDimmer = (
                    rect.top <= 0 && rect.left <= 0 &&
                    rect.width >= window.innerWidth && rect.height >= window.innerHeight &&
                    parseFloat(style.opacity) < 0.9 && parseFloat(style.opacity) > 0.1 // Often semi-transparent
                );

                const isLargeObstructingElement = (
                    (rect.width >= window.innerWidth * 0.7 && rect.height >= window.innerHeight * 0.5) || // Covers large area
                    (rect.width >= window.innerWidth * 0.9 && rect.height > 50) || // Wide banner
                    (rect.height >= window.innerHeight * 0.9 && rect.width > 50)    // Tall banner
                );

                if (zIndex > 100) { // High z-index is a strong indicator for overlays
                    if (isFullScreenDimmer || isLargeObstructingElement) {
                        tryRemove(el, `fixed/sticky high z-index: ${zIndex}`);
                    } else if (rect.width > 50 && rect.height > 50) { // Avoid removing small fixed items unless clearly popups
                        // Check if it has close buttons or typical popup text
                        const text = (el.textContent || '').toLowerCase();
                        const hasClose = el.querySelector('[class*="close"], [aria-label*="close"], [title*="close"]');
                        if (hasClose || text.includes('subscribe') || text.includes('chat') || text.includes('support')) {
                           // More targeted removal for smaller fixed items
                           if (rect.bottom > window.innerHeight - 150 && rect.right > window.innerWidth - 150 && rect.width < 400 && rect.height < 600) { // common chat widget area
                               tryRemove(el, `fixed/sticky chat/support like: ${zIndex}`);
                           }
                        }
                    }
                }
            }

            // Check for elements that disable scrolling on body/html and are on top
             if ((document.body.style.overflow === 'hidden' || document.documentElement.style.overflow === 'hidden' ||
                 window.getComputedStyle(document.body).overflow === 'hidden' || window.getComputedStyle(document.documentElement).overflow === 'hidden') &&
                 zIndex > 10 && position === 'fixed' &&
                 rect.width >= window.innerWidth && rect.height >= window.innerHeight
             ) {
                 if (el.tagName.toLowerCase() !== 'body' && el.tagName.toLowerCase() !== 'html') {
                    tryRemove(el, 'full screen fixed overlay with body scroll hidden');
                 }
             }
        } catch (e) { /* console.error('Error processing element for fixed/sticky:', el, e); */ }
    });

    elementsToRemove.forEach(el => {
        if (el && el.parentNode) {
            el.remove();
            removedCount++;
        }
    });
    // console.log(`Removed ${removedCount} elements from this frame.`);

    // Restore body/html overflow if it was set to hidden by a popup
    const restoreScroll = (doc) => {
        if (doc.body && (doc.body.style.overflow === 'hidden' || window.getComputedStyle(doc.body).overflow === 'hidden')) {
            doc.body.style.overflow = 'auto';
        }
        if (doc.documentElement && (doc.documentElement.style.overflow === 'hidden' || window.getComputedStyle(doc.documentElement).overflow === 'hidden')) {
            doc.documentElement.style.overflow = 'auto';
        }
    };
    restoreScroll(document); // For current document (frame)

    // If this is the main window, try to ensure its scroll is also restored
    // (though popups in iframes shouldn't affect main window scroll directly unless they use JS to do so)
    if (window.top === window.self) {
        restoreScroll(window.top.document);
    }
    return removedCount; // Return how many elements were removed
}
"""


async def execute_popup_remover_on_frame(frame: Frame):
    """Executes the popup remover script on a given frame."""
    try:
        # print(f"Attempting to remove popups in frame: {frame.url}")
        removed_count = await frame.evaluate(POPUP_REMOVER_SCRIPT)
        # print(f"Removed {removed_count} elements from frame: {frame.url}")
        # Recursively apply to child frames
        for child_frame in frame.child_frames:
            await execute_popup_remover_on_frame(child_frame)
    except Exception as e:
        # Errors here can happen if frame navigates away or is detached
        print(
            f"Could not execute script in frame {frame.url or '[no url]'}: {e}",
            file=sys.stderr,
        )


async def remove_all_popups_and_iframes_content(page: Page):
    """
    Removes popups from the main page and all its iframes.
    Optionally, could also attempt to remove problematic iframes themselves (e.g., ad iframes).
    For now, focuses on cleaning content *within* frames.
    """
    # First, run on the main frame
    await execute_popup_remover_on_frame(page.main_frame)
    # The above recursive call should handle all nested frames.

    # Additionally, one could try to identify and remove entire iframes that are problematic (e.g., ads)
    # This is more aggressive and needs careful heuristics.
    # For example:
    await page.evaluate(
        """() => {
      document.querySelectorAll('iframe').forEach(iframe => {
        const src = iframe.src || "";
        const name = iframe.name || "";
        const id = iframe.id || "";
        // Keywords for ad-related iframes
        const adKeywords = ['ads', 'adserver', 'doubleclick', 'googleads', 'googlesyndication', 'adform', 'yieldbird'];
        if (adKeywords.some(kw => src.includes(kw) || name.includes(kw) || id.includes(kw))) {
          // Check size, if it's small and looks like an ad banner
          const rect = iframe.getBoundingClientRect();
          if (rect.width > 50 && rect.height > 50 && rect.width < 800 && rect.height < 300) { // Example dimensions
            console.log('Removing ad iframe:', iframe);
            iframe.remove();
          }
        }
      });
    }"""
    )


async def setup_browser_context(
    playwright: Playwright,
) -> Browser:  # Return type is Browser
    """Sets up and launches the browser."""
    browser = await playwright.chromium.launch(
        headless=True,  # Set to False for debugging
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",  # Often needed with --no-sandbox
            "--disable-dev-shm-usage",  # Important for Docker/CI environments
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--disable-popup-blocking",  # We handle popups, but this can prevent some native ones
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-component-extensions-with-background-pages",  # Might help
            "--disable-features=IsolateOrigins,site-per-process,TranslateUI,OptimizationHints,MediaRouter",
            "--disable-web-security",  # Use with caution, can be a security risk if you process untrusted sites and do more than screenshots
            "--disable-site-isolation-trials",
            "--no-first-run",
            "--no-default-browser-check",
            "--window-size=1920,1080",  # Set a consistent window size
            # Removed some args that might be too aggressive or not directly related to popups
            # '--disable-window-activation',
            # '--disable-focus-on-load',
            # '--no-startup-window',
            # '--window-position=0,0',
        ],
    )
    return browser


async def simulate_human_behavior(page: Page):
    """Simulates some human-like interactions."""
    await random_sleep(0.2, 0.8)
    try:
        viewport_size = page.viewport_size
        if viewport_size:
            await page.mouse.move(
                random.randint(
                    int(viewport_size["width"] * 0.1), int(viewport_size["width"] * 0.9)
                ),
                random.randint(
                    int(viewport_size["height"] * 0.1),
                    int(viewport_size["height"] * 0.9),
                ),
            )
            await random_sleep(0.1, 0.3)
            # Gentle scroll, can help trigger lazy-loaded elements or sticky headers to behave
            scroll_amount = random.randint(100, 300)
            await page.mouse.wheel(0, scroll_amount)
            await random_sleep(0.2, 0.5)
            await page.mouse.wheel(0, -scroll_amount)  # Scroll back up slightly
    except Exception as e:
        print(f"Error during human behavior simulation: {e}", file=sys.stderr)


async def process_url(
    node: Node, browser: Browser, save_dir: str
) -> Optional[Dict[str, Any]]:
    """处理单个URL的函数，成功返回 None，失败返回错误信息字典"""
    context: Optional[BrowserContext] = None  # Type hint for clarity
    page: Optional[Page] = None
    try:
        user_agent = random.choice(USER_AGENTS)
        context = await browser.new_context(
            user_agent=user_agent,
            is_mobile=False,  # Consider parameterizing if needed
            has_touch=False,
            java_script_enabled=True,
            locale="en-US",  # Or make this configurable
            timezone_id="Europe/London",  # Or make this configurable
            permissions=[
                "notifications"
            ],  # 'geolocation' could also be denied or prompted if it causes popups
            bypass_csp=True,  # May help with script injection in some cases, use cautiously
            viewport={"width": 1920, "height": 1080},  # Set consistent viewport
            # device_scale_factor=1, # Ensure no scaling issues
        )
        await context.add_cookies(
            [
                # Example: Pre-accepting a common cookie consent cookie if you know its name for certain sites.
                # {'name': 'cookie_consent', 'value': 'accepted', 'domain': '.example.com', 'path': '/'}
            ]
        )  # Can be used to pre-set cookies, e.g., for consent

        page = await context.new_page()

        # Abort requests for images, fonts, videos, stylesheets to speed up loading
        # and reduce chances of external resources interfering.
        # Keep stylesheets if visual accuracy is paramount and they don't cause issues.
        # For this task, removing popups, stylesheets are likely important for layout.
        # Fonts can be heavy, aborting them might be fine.
        await page.route(
            "**/*.{png,jpg,jpeg,gif,svg,webp,ico,woff,woff2,ttf,eot,mp4,webm}",
            lambda route: route.abort(),
        )
        # await page.route("**/*", lambda route: print(f"Request: {route.request.method} {route.request.url}") or route.continue_())

        # Set up dialog handler
        page.on("dialog", handle_dialog)

        # Enhanced script to run on new document creation (main or iframes)
        # This helps disable alerts/prompts immediately
        await page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined }); // Attempt to hide webdriver
            window.alert = function(msg) { console.log('Alert suppressed:', msg); };
            window.confirm = function(msg) { console.log('Confirm suppressed:', msg); return true; };
            window.prompt = function(msg, defaultVal) { console.log('Prompt suppressed:', msg); return null; };
            // window.onbeforeunload = function() {}; // This can prevent legitimate navigations if too broad.
                                                    // If used, ensure it's what you want.

            // Try to prevent some common anti-bot measures or popup triggers
            // Object.defineProperty(document, 'visibilityState', { get: () => 'visible' });
            // Object.defineProperty(document, 'hidden', { get: () => false });
            // document.addEventListener('visibilitychange', e => e.stopImmediatePropagation(), true);
        """
        )

        try:
            response = await page.goto(
                url=node.url, wait_until="domcontentloaded", timeout=45000
            )  # domcontentloaded can be faster
        except Exception as e:  # Catch navigation errors specifically
            error_msg = f"Failed to navigate to {node.url}: {str(e)}"
            if page:
                await page.close()
            if context:
                await context.close()
            return {
                "url": node.url,
                "error": error_msg,
                "node_id": getattr(node, "id", "N/A"),
            }

        if not response:  # Should be caught by exception above, but as a safeguard
            error_msg = "Failed to load: No response object"
            if page:
                await page.close()
            if context:
                await context.close()
            return {
                "url": node.url,
                "error": error_msg,
                "node_id": getattr(node, "id", "N/A"),
            }

        if response.status >= 400:
            error_msg = f"Failed to load: Status {response.status}"
            if page:
                await page.close()
            if context:
                await context.close()
            return {
                "url": node.url,
                "error": error_msg,
                "node_id": getattr(node, "id", "N/A"),
            }

        await random_sleep(0.5, 1.5)  # Wait a bit for initial scripts to run

        # First pass of popup removal
        print(f"[{node.url}] Running initial popup removal...")
        await remove_all_popups_and_iframes_content(page)
        await random_sleep(0.3, 0.7)  # Wait for any dynamic changes after first removal

        # Simulate some interaction
        await simulate_human_behavior(page)
        await random_sleep(
            0.5, 1.0
        )  # Wait after interaction, some popups appear on scroll/mouse move

        # Second pass of popup removal (in case interaction triggered new ones)
        print(f"[{node.url}] Running second popup removal pass...")
        await remove_all_popups_and_iframes_content(page)
        await random_sleep(0.2, 0.5)

        # Take screenshot
        screenshot_filename = f"{node.url.replace('https://', '').replace('http://', '').strip('/').replace('/', '_')}.png"
        screenshot_filename = re.sub(
            r'[<>:"/\\|?*]', "_", screenshot_filename
        )  # Sanitize
        screenshot_filename = re.sub(
            r"_+", "_", screenshot_filename
        )  # Consolidate multiple underscores
        screenshot_filename = screenshot_filename.strip("_")

        max_len = 200
        if len(screenshot_filename) > max_len:
            # Try to preserve domain and truncate path intelligently
            try:
                from urllib.parse import urlparse

                parsed_url = urlparse(node.url)
                domain = parsed_url.netloc.replace("www.", "")
                path = parsed_url.path.strip("/").replace("/", "_")
                filename_base = f"{domain}_{path}" if path else domain
                filename_base = re.sub(r'[<>:"/\\|?*]', "_", filename_base)
                filename_base = re.sub(r"_+", "_", filename_base).strip("_")

                if len(filename_base) > max_len - 4:  # -4 for ".png"
                    # Truncate from the end of the path part if possible
                    domain_part = domain
                    allowed_path_len = (
                        max_len - len(domain_part) - 1 - 4
                    )  # -1 for underscore
                    if allowed_path_len > 10:  # Keep at least some path
                        path_part = path[:allowed_path_len]
                        screenshot_filename = f"{domain_part}_{path_part}.png"
                    else:  # Domain itself is too long or no space for path
                        screenshot_filename = f"{domain_part[:max_len-4]}.png"
                else:
                    screenshot_filename = f"{filename_base}.png"

            except Exception:  # Fallback to original simpler truncation
                parts = screenshot_filename.split("_", 1)
                domain_part = parts[0]
                path_part_original = parts[1] if len(parts) > 1 else ""
                allowed_path_len = (
                    max_len - len(domain_part) - 1 - 4
                )  # for underscore and .png
                if allowed_path_len > 10:
                    screenshot_filename = (
                        f"{domain_part}_{path_part_original[:allowed_path_len]}.png"
                    )
                else:
                    screenshot_filename = f"{domain_part[:max_len-4]}.png"

        screenshot_path = os.path.join(save_dir, screenshot_filename)
        node.snapshot_path = screenshot_path

        # Ensure save_dir exists
        os.makedirs(save_dir, exist_ok=True)

        print(f"[{node.url}] Taking screenshot: {screenshot_path}")
        await page.screenshot(
            path=screenshot_path, full_page=True, timeout=30000
        )  # Increased timeout for screenshot

        await page.close()
        page = None
        await context.close()
        context = None
        await random_sleep(0.1, 0.3)
        return None  # Success

    except Exception as e:
        error_msg = (
            f"Unexpected error processing {node.url}: {type(e).__name__} - {str(e)}"
        )
        import traceback

        print(
            f"{error_msg}\n{traceback.format_exc()}", file=sys.stderr
        )  # Print full traceback for debugging
        if page:
            try:
                await page.close()
            except Exception as close_e:
                print(f"Error closing page for {node.url}: {close_e}", file=sys.stderr)
        if context:
            try:
                await context.close()
            except Exception as close_e:
                print(
                    f"Error closing context for {node.url}: {close_e}", file=sys.stderr
                )
        return {
            "url": node.url,
            "error": error_msg,
            "node_id": getattr(node, "id", "N/A"),
        }


async def process_urls(
    nodes: List[Node], save_dir: str, max_concurrent: int = 15
) -> List[Dict[str, Any]]:
    """并行处理多个URL，返回错误列表"""
    errors = []  # 用于收集错误
    async with async_playwright() as p:
        browser = await setup_browser_context(p)

        # 使用信号量限制并发数
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_process_url(node):  # 修改参数为 node
            async with semaphore:
                return await process_url(
                    node, browser, save_dir
                )  # 调用修改后的 process_url

        # 创建所有任务
        tasks = [bounded_process_url(node) for node in nodes]

        print(f"Processing {len(tasks)} URLs...")  # 添加提示信息
        for f in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Processing {os.path.basename(save_dir)}",
        ):
            result = await f
            if result:  # 如果 process_url 返回了错误字典
                errors.append(result)

        await browser.close()
    return errors  # 返回收集到的错误列表


async def main():
    # 定义输入和输出路径
    json_input_dir = "/home/zyy/web_analyzer/webvoyager_output/URL_json"
    snapshot_output_dir = "/home/zyy/web_analyzer/webvoyager_output/snapshots"
    error_log_file = "/home/zyy/web_analyzer/webvoyager_output/snapshot_errors.log"  # 定义日志文件路径

    # 查找所有 JSON 文件 (如果你只想处理单个文件，恢复原来的 file_paths 列表)
    file_paths = glob.glob(os.path.join(json_input_dir, "*.json"))
    file_paths = [
        "/home/zyy/url_process/webwalker_urls/url_tree_www.cs.zju.edu.cn_csen.json"
    ]
    if not file_paths:
        print(f"No JSON files found in {json_input_dir}", file=sys.stderr)
        return

    print(f"Found {len(file_paths)} JSON files to process.")

    all_run_errors = []  # 收集本次运行的所有错误

    for file_path in file_paths:
        print(f"\nProcessing file: {file_path}")
        try:
            tree = Tree.load_from_json(file_path)
            # 确保 tree.all_nodes 存在且可用
            if not hasattr(tree, "all_nodes") or not tree.all_nodes:
                msg = f"Warning: Skipping file {file_path}. Cannot find or access 'all_nodes'."
                print(msg, file=sys.stderr)
                all_run_errors.append(
                    {
                        "source_file": file_path,
                        "url": "N/A (File Level Error)",
                        "error": "Cannot find or access all_nodes",
                    }
                )
                continue

            all_nodes = tree.select_unrepeated_nodes(use_title=True)

            # 创建对应的快照保存目录
            base_filename = os.path.basename(file_path)
            save_dir_name = base_filename.replace(".json", "")
            save_dir = os.path.join(snapshot_output_dir, save_dir_name)
            os.makedirs(save_dir, exist_ok=True)

            # 处理 URL 并收集错误
            file_errors = await process_urls(all_nodes, save_dir)

            if file_errors:
                print(
                    f"Encountered {len(file_errors)} errors while processing {file_path}. Check log file for details."
                )
                # 为错误添加来源文件信息
                for error in file_errors:
                    error["source_file"] = file_path
                all_run_errors.extend(file_errors)

            # 即使有错误，也保存（可能部分更新了 snapshot_path 的）tree
            tree.save_to_json(file_path)

        except Exception as e:
            error_info = {
                "source_file": file_path,
                "url": "N/A (File Level Error)",
                "error": f"Failed to process file: {str(e)}",
            }
            error_message = f"Critical error processing file {file_path}: {str(e)}"
            print(f"\n{error_message}", file=sys.stderr)
            all_run_errors.append(error_info)

    # 运行结束后，将所有收集到的错误写入日志文件
    if all_run_errors:
        print(f"\nTotal errors encountered during the run: {len(all_run_errors)}")
        try:
            with open(error_log_file, "w", encoding="utf-8") as f:
                f.write(
                    f"Snapshot Processing Errors - Run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(
                    "=============================================================\n\n"
                )
                for error in all_run_errors:
                    f.write(f"Source File: {error.get('source_file', 'Unknown')}\n")
                    f.write(f"Node ID: {error.get('node_id', 'N/A')}\n")
                    f.write(f"URL: {error.get('url', 'N/A')}\n")
                    f.write(f"Error: {error.get('error', 'Unknown error')}\n")
                    f.write("-" * 20 + "\n")
            print(f"Error details saved to {error_log_file}")
        except Exception as log_e:
            print(
                f"Failed to write error log file {error_log_file}: {log_e}",
                file=sys.stderr,
            )
    else:
        print("\nRun completed successfully with no errors reported.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as global_e:
        print(
            f"\nAn unexpected error occurred at the top level: {global_e}",
            file=sys.stderr,
        )
        try:
            error_log_file = "/home/zyy/web_analyzer/webvoyager_output/snapshot_errors.log"  # 确保路径一致
            with open(error_log_file, "a", encoding="utf-8") as f:  # 使用追加模式 'a'
                f.write("\n====================\n")
                f.write("TOP LEVEL EXCEPTION:\n")
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {global_e}\n")
                f.write("====================\n")
        except Exception as log_final_e:
            print(
                f"Additionally failed to log the top level error: {log_final_e}",
                file=sys.stderr,
            )

        sys.exit(1)

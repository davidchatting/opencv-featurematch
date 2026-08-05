// Regenerates screenshot.png from the live p5js/ demo (run by CI after
// p5js/ is synced from the editor sketch, so it always reflects that
// sketch's current state — see .github/workflows/build.yml).
//
// Waits on the sketch's own top-level `result` variable (set once
// alignImages() resolves) rather than a fixed delay, so it's not flaky
// under CI load.
const { chromium } = require('playwright');
const path = require('path');

const DEMO_URL = process.argv[2] || 'http://localhost:8099/';
const OUT_PATH = path.join(__dirname, '..', 'screenshot.png');

(async () => {
  const browser = await chromium.launch({ args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 800, height: 400 } });

  await page.goto(DEMO_URL, { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForFunction(
    () => typeof result !== 'undefined' && result && result.valid !== undefined,
    { timeout: 30000 }
  );
  await page.waitForTimeout(300); // let the frame with the result actually paint

  const valid = await page.evaluate(() => result.valid);
  if (!valid) {
    await browser.close();
    throw new Error('Demo sketch ran but result.valid was false — not updating screenshot.png');
  }

  await page.screenshot({ path: OUT_PATH });
  console.log(`Wrote ${OUT_PATH}`);
  await browser.close();
})();

/* Render and interact with the self-contained report in a real Chrome page.
 * This writes report-only validation derivatives and does not open any artifact
 * for mutation.  Run after build_report.py.
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {spawn} = require('child_process');

const root = __dirname;
const report = path.join(root, 'retry20_candidate_diagnostic_advisor_dataset_report.html');
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

async function main() {
  const profile = fs.mkdtempSync('/tmp/retry20-report-chrome-');
  const chrome = spawn('/usr/bin/google-chrome', [
    '--headless=new', '--no-sandbox', '--enable-webgl', '--ignore-gpu-blocklist',
    '--use-angle=swiftshader', '--disable-dev-shm-usage', '--remote-debugging-port=0',
    '--user-data-dir=' + profile, 'about:blank',
  ], {stdio: 'ignore'});
  let ws;
  try {
    let port;
    for (let i = 0; i < 60; i++) {
      try { port = fs.readFileSync(profile + '/DevToolsActivePort', 'utf8').split('\n')[0]; break; } catch { await wait(100); }
    }
    if (!port) throw new Error('Chrome DevTools port did not appear');
    const target = await (await fetch('http://127.0.0.1:' + port + '/json/new?about:blank', {method: 'PUT'})).json();
    ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
    let id = 0;
    const waiting = new Map();
    const exceptions = [], consoleErrors = [];
    ws.onmessage = event => {
      const message = JSON.parse(event.data);
      if (message.id) { const resolve = waiting.get(message.id); if (resolve) resolve(message); waiting.delete(message.id); return; }
      if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails);
      if (message.method === 'Runtime.consoleAPICalled' && ['error', 'assert'].includes(message.params.type)) consoleErrors.push(message.params);
    };
    const call = (method, params = {}) => new Promise(resolve => { const next = ++id; waiting.set(next, resolve); ws.send(JSON.stringify({id: next, method, params})); });
    const evaluate = async expression => {
      const result = await call('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
      if (result.result?.exceptionDetails) throw new Error(JSON.stringify(result.result.exceptionDetails));
      return result.result.result.value;
    };
    const screenshot = async name => {
      const image = await call('Page.captureScreenshot', {format: 'png'});
      fs.writeFileSync(path.join(root, name), Buffer.from(image.result.data, 'base64'));
    };
    await call('Runtime.enable'); await call('Page.enable');
    await call('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
    await call('Page.navigate', {url: 'file://' + report});
    await wait(1800);
    const checks = {};
    checks.loaded = await evaluate('typeof Plotly !== "undefined" && D.tracks.length === 16 && document.querySelectorAll("script[src]").length === 0');
    checks.static_content = await evaluate('document.body.innerText.includes("root-connectivity 未重新认证") && document.body.innerText.includes("29.684") && document.body.innerText.includes("46.73264")');

    await evaluate('$("coverage").scrollIntoView({block:"center"})'); await wait(1300);
    checks.coverage = await evaluate('({traces:$("coverage").data.map(x=>x.name), canvases:$("coverage").querySelectorAll("canvas").length, hasRobot:$("coverage").data.some(x=>x.name==="机器人 beta=0"), hasL0:$("coverage").data.some(x=>x.name.startsWith("L0")), hasL1:$("coverage").data.some(x=>x.name.startsWith("L1")), symbols:$("coverage").data.filter(x=>x.name.startsWith("L")).map(x=>x.marker.symbol)})');
    await screenshot('coverage_validation.png');
    await evaluate('$("coverageLocal").click()'); await wait(500);
    checks.coverage_angle = await evaluate('$("coverage").layout.title.text.includes("非裁剪") && $("coverage").data.some(x=>x.name==="机器人 beta=0")');

    await evaluate('$("tracking").scrollIntoView({block:"center"}); trackingVisible=true; tracking()'); await wait(1300);
    checks.tracking_global = await evaluate('({id:current().id,traces:$("tracking").data.map(x=>x.name),canvases:$("tracking").querySelectorAll("canvas").length,teacherWidth:$("tracking").data.find(x=>x.name==="Teacher").line.width,targetDash:$("tracking").data.find(x=>x.name==="Target").line.dash,zero:$("tracking").data.some(x=>x.name==="机器人 beta=0")})');
    await screenshot('tracking_global_validation.png');
    await evaluate('$("trajectory").value="retry19_exact_seam_rectangle_2";$("trajectory").dispatchEvent(new Event("change"));$("trackingLocal").click();$("projection").value="yz";$("projection").dispatchEvent(new Event("change"))'); await wait(700);
    checks.tracking_local = await evaluate('({id:current().id,local:state.trackingLocal,projection:state.projection,zero:$("tracking").data.some(x=>x.name==="机器人 beta=0"),teacherMax:current().metrics.Teacher.max_beta_step_deg,betaNames:$("beta").data.map(x=>x.name),betaMax:Math.max(...$("beta").data[0].y.filter(Number.isFinite)),rawBetaMax:Math.max(...$("beta").data[1].y.filter(Number.isFinite))})');
    await screenshot('tracking_local_validation.png');
    await evaluate('$("showL0raw").click();$("showL0dls").click();$("showL1raw").click();$("showL1dls").click()'); await wait(600);
    checks.toggles = await evaluate('$("tracking").data.map(x=>x.name)');
    await evaluate('$("trackingGlobal").click();$("theme").click()'); await wait(600);
    checks.theme = await evaluate('({dark:document.documentElement.dataset.theme==="dark",trackingGlobal:$("tracking").layout.title.text.includes("全局"),overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth})');

    await call('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: false});
    await evaluate('window.dispatchEvent(new Event("resize")); ["loss","beta"].forEach(id=>Plotly.Plots.resize($(id)))'); await wait(700);
    checks.narrow = await evaluate('(()=>{const tables=[...document.querySelectorAll("table")].map(e=>{const p=e.parentElement,r=e.getBoundingClientRect(),pr=p.getBoundingClientRect();return {tableRight:Math.round(r.right),parentClass:p.className,parentRight:Math.round(pr.right),parentClient:p.clientWidth,parentScroll:p.scrollWidth,parentOverflow:getComputedStyle(p).overflowX}});return {client:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth,title:document.title,plots:["coverage","loss","tracking","beta"].map(id=>{const e=$(id),r=e.getBoundingClientRect();return {id,scroll:e.scrollWidth,client:e.clientWidth,left:Math.round(r.left),right:Math.round(r.right)}}),tables}})()');
    await screenshot('narrow_validation.png');
    const pass = checks.loaded && checks.static_content && checks.coverage.canvases > 0 && checks.coverage.hasRobot && checks.coverage.hasL0 && checks.coverage.hasL1 && new Set(checks.coverage.symbols).size >= 3 && checks.coverage_angle && checks.tracking_global.id === 'fixed_circle_r40_u100' && checks.tracking_global.canvases > 0 && checks.tracking_global.zero && checks.tracking_global.teacherWidth >= 8 && checks.tracking_global.targetDash === 'dash' && checks.tracking_local.id === 'retry19_exact_seam_rectangle_2' && checks.tracking_local.local && checks.tracking_local.projection === 'yz' && checks.tracking_local.zero && checks.tracking_local.teacherMax > 29 && checks.tracking_local.betaNames.join('|') === 'Teacher max |Δβ||L1 raw max |Δβ||7° reference' && checks.tracking_local.betaMax > 29 && checks.tracking_local.rawBetaMax < 1 && checks.toggles.includes('L0 raw Student') && checks.toggles.includes('L0 DLS2') && !checks.toggles.includes('L1 raw Student') && !checks.toggles.includes('L1 DLS2') && checks.theme.dark && checks.theme.trackingGlobal && !checks.theme.overflow && checks.narrow.client === checks.narrow.scroll && exceptions.length === 0 && consoleErrors.length === 0;
    fs.writeFileSync(path.join(root, 'browser_validation.json'), JSON.stringify({
      observed_at: new Date().toISOString(), passed: pass, report_sha256: hash(report), checks, exceptions, consoleErrors,
      engine: 'Google Chrome headless with WebGL SwiftShader',
      boundary: 'HTML rendering and interaction verification only; it is not an artifact rehash or scientific Gate.',
    }, null, 2) + '\n');
    console.log(JSON.stringify({passed: pass, checks, exceptions: exceptions.length, consoleErrors: consoleErrors.length}));
    if (!pass) process.exitCode = 1;
  } finally {
    if (ws) ws.close();
    chrome.kill('SIGTERM');
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });

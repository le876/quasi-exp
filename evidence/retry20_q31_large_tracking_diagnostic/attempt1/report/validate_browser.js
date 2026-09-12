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
    checks.loaded = await evaluate('typeof Plotly !== "undefined" && D.tracks.length === 19 && document.querySelectorAll("script[src]").length === 0');
    checks.defaultLarge = await evaluate("$('trackGroup').value==='q31_large'&&current().id==='large_star'&&$('trajectory').options.length===3");
    checks.static_content = await evaluate('document.body.innerText.includes("root-connectivity 未重新认证") && document.body.innerText.includes("29.684") && document.body.innerText.includes("46.73264")');

    await evaluate('$("coverage").scrollIntoView({block:"center"})'); await wait(1300);
    checks.coverage = await evaluate('({traces:$("coverage").data.map(x=>x.name), canvases:$("coverage").querySelectorAll("canvas").length, hasRobot:$("coverage").data.some(x=>x.name==="机器人 beta=0"), hasL0:$("coverage").data.some(x=>x.name.startsWith("L0")), hasL1:$("coverage").data.some(x=>x.name.startsWith("L1")), symbols:$("coverage").data.filter(x=>x.name.startsWith("L")).map(x=>x.marker.symbol)})');
    await screenshot('coverage_validation.png');
    await evaluate('$("coverageLocal").click()'); await wait(500);
    checks.coverage_angle = await evaluate('$("coverage").layout.title.text.includes("非裁剪") && $("coverage").data.some(x=>x.name==="机器人 beta=0")');

    await evaluate("$('trackGroup').value='original';$('trackGroup').dispatchEvent(new Event('change'))");
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


    // Compare rendered legend text against actual trace colors, not a second palette.
    const inspectLegend = id => evaluate(`(() => {
      const plot = document.getElementById(${JSON.stringify(id)});
      const normalize = c => {const e=document.createElement('span');e.style.color=c;document.body.append(e);const v=getComputedStyle(e).color;e.remove();return v};
      const luminance = c => {const v=c.match(/[0-9.]+/g).slice(0,3).map(Number).map(v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4});return .2126*v[0]+.7152*v[1]+.0722*v[2]};
      return [...document.querySelectorAll('#'+plot.id+'-legend button')].map(button=>{
        const trace=plot.data[Number(button.dataset.index)],style=getComputedStyle(button),actual=style.color,expected=normalize(trace.line?.color||trace.marker?.color);
        const a=luminance(actual),b=luminance(style.backgroundColor),contrast=(Math.max(a,b)+.05)/(Math.min(a,b)+.05),r=button.getBoundingClientRect();
        return {name:trace.name,actual,expected,size:parseFloat(style.fontSize),contrast,withinViewport:r.left>=0&&r.right<=document.documentElement.clientWidth,textFits:button.scrollWidth<=button.clientWidth+1,pressed:button.getAttribute('aria-pressed'),pass:actual===expected&&parseFloat(style.fontSize)>=14&&contrast>=4.5&&r.left>=0&&r.right<=document.documentElement.clientWidth&&button.scrollWidth<=button.clientWidth+1};
      });
    })()`);
    checks.legends = {};
    // The preceding test leaves the page in dark mode. Check all plots in both themes.
    for (const theme of ['dark','light']) {
      await evaluate(`if(document.documentElement.dataset.theme!==${JSON.stringify(theme)}) $('theme').click()`);
      for (const id of ['coverage','loss','tracking','beta']) {
        await evaluate(`$('${id}').scrollIntoView({block:'center'})`); await wait(600);
        checks.legends[theme+'_'+id] = await inspectLegend(id);
      }
    }
    await evaluate(`$('tracking').scrollIntoView({block:'center'});$('trajectory').selectedIndex=0;$('trajectory').dispatchEvent(new Event('change'))`);await wait(500);
    const navState = () => evaluate(`({index:$('trajectory').selectedIndex,position:$('trajectory-position').textContent,previousDisabled:$('trajectory-previous').disabled,nextDisabled:$('trajectory-next').disabled,title:$('tracking').layout.title.text,metrics:$('trackMetrics').textContent.includes(current().id),teacherMatches:JSON.stringify($('tracking').data.find(t=>t.name==='Teacher').x)===JSON.stringify((current().closed?[...current().teacher,current().teacher[0]]:current().teacher).map(p=>p?.[0]??null)),betaMatches:JSON.stringify($('beta').data[0].y)===JSON.stringify(maxJointSteps(current().teacherBeta,current().closed)),id:current().id})`);
    checks.navigation = {first:await navState()};
    await evaluate(`$('trajectory-next').click()`);await wait(400);checks.navigation.next=await navState();
    await evaluate(`$('trajectory-previous').click()`);await wait(400);checks.navigation.previous=await navState();
    await evaluate(`$('trajectory').selectedIndex=15;$('trajectory').dispatchEvent(new Event('change'));$('trajectory-next').click()`);await wait(400);checks.navigation.last=await navState();
    await evaluate(`$('trajectory-previous').click()`);await wait(400);checks.navigation.fromLast=await navState();
    const n=checks.navigation;
    checks.navigationPassed=n.first.index===0&&n.first.previousDisabled&&n.first.position==='1 / 16'&&n.next.index===1&&n.next.position==='2 / 16'&&n.previous.index===0&&n.last.index===15&&n.last.nextDisabled&&n.last.position==='16 / 16'&&n.fromLast.index===14&&Object.values(n).every(v=>v.metrics&&v.teacherMatches&&v.betaMatches&&v.title.includes(v.id));
    await evaluate(`document.querySelector('#tracking-legend button').click()`);await wait(350);
    checks.legendToggle=await evaluate(`$('tracking').data[0].visible==='legendonly'&&document.querySelector('#tracking-legend button').getAttribute('aria-pressed')==='false'`);
    await evaluate(`document.querySelector('#tracking-legend button').focus();document.querySelector('#tracking-legend button').click()`);await wait(350);
    checks.legendToggle=checks.legendToggle&&await evaluate(`$('tracking').data[0].visible===true&&document.querySelector('#tracking-legend button').getAttribute('aria-pressed')==='true'&&document.activeElement===document.querySelector('#tracking-legend button')`);
    checks.inlineLegend = await evaluate(`(()=>{ReportPlot.syncLabels();return [...document.querySelectorAll('[data-series]')].map(el=>{const probe=document.createElement('span');probe.style.color=ReportPlot.palette()[el.dataset.series];document.body.append(probe);const expected=getComputedStyle(probe).color;probe.remove();return {series:el.dataset.series,pass:getComputedStyle(el).color===expected&&parseFloat(getComputedStyle(el).fontSize)>=14}})})()`);
    await screenshot('legend_navigation_validation.png');

    await call('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: false});
    await evaluate('window.dispatchEvent(new Event("resize")); ["loss","beta"].forEach(id=>Plotly.Plots.resize($(id)))'); await wait(700);
    checks.narrow = await evaluate('(()=>{const tables=[...document.querySelectorAll("table")].map(e=>{const p=e.parentElement,r=e.getBoundingClientRect(),pr=p.getBoundingClientRect();return {tableRight:Math.round(r.right),parentClass:p.className,parentRight:Math.round(pr.right),parentClient:p.clientWidth,parentScroll:p.scrollWidth,parentOverflow:getComputedStyle(p).overflowX}});return {client:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth,title:document.title,plots:["coverage","loss","tracking","beta"].map(id=>{const e=$(id),r=e.getBoundingClientRect();return {id,scroll:e.scrollWidth,client:e.clientWidth,left:Math.round(r.left),right:Math.round(r.right)}}),tables}})()');
    checks.narrowLegends = [];
    for (const id of ['coverage','loss','tracking','beta']) {
      await evaluate(`$('${id}').scrollIntoView({block:'center'})`); await wait(600);
      checks.narrowLegends.push(...await inspectLegend(id));
    }
    checks.legendPassed = Object.values(checks.legends).every(rows=>rows.length>0&&rows.every(r=>r.pass))&&checks.narrowLegends.length>0&&checks.narrowLegends.every(r=>r.pass)&&checks.inlineLegend.length>0&&checks.inlineLegend.every(r=>r.pass)&&checks.legendToggle;
    await evaluate(`$('trajectory').scrollIntoView({block:'center'})`); await wait(300);
    checks.narrowControls=await evaluate(`({scrollX:window.scrollX,pass:[...document.querySelectorAll('.toolbar select,.path-navigation button')].every(e=>{const r=e.getBoundingClientRect();return r.left>=0&&r.right<=document.documentElement.clientWidth})})`);
    await screenshot('narrow_validation.png');
    await call('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await evaluate("$('showL0raw').checked=false;$('showL0dls').checked=false;$('showL1raw').checked=true;$('showL1dls').checked=true;$('trackGroup').value='q31_large';$('trackGroup').dispatchEvent(new Event('change'));$('tracking').scrollIntoView({block:'center'})");await wait(800);
    checks.largePaths=[];
    for (let i=0;i<3;i++) {
      await evaluate(`$('trajectory').selectedIndex=${i};$('trajectory').dispatchEvent(new Event('change'))`);await wait(650);
      checks.largePaths.push(await evaluate(`(()=>{const t=current(),g=D.largeSuite.tracks.find(g=>g.id===t.id),plot=$('tracking'),target=plot.data.find(p=>p.name==='Target');return {id:t.id,count:t.count,closed:t.closed,position:$('trajectory-position').textContent,coordinates:JSON.stringify(t.target)===JSON.stringify(g.waypoints_m.map(p=>p.map(v=>v*1000))),drawnPoints:target.x.length,expectedDrawnPoints:t.count+(t.closed?1:0),zero:plot.data.some(p=>p.name==='机器人 beta=0'),metrics:$('trackMetrics').textContent.includes('retry20独立新评估'),teacherMissing:t.teacher.filter(p=>p.some(v=>v===null)).length}})()`));
      await screenshot('q31_large_'+i+'_validation.png');
    }
    checks.largePathsPassed=checks.defaultLarge&&checks.largePaths.length===3&&checks.largePaths.every(p=>p.coordinates&&p.drawnPoints===p.expectedDrawnPoints&&p.zero&&p.metrics)&&checks.largePaths.reduce((n,p)=>n+p.count,0)===1504&&checks.largePaths.filter(p=>p.closed).length===1;
    const pass = checks.largePathsPassed && checks.narrowControls.pass && checks.narrowControls.scrollX === 0 && checks.navigationPassed && checks.legendPassed && checks.loaded && checks.static_content && checks.coverage.canvases > 0 && checks.coverage.hasRobot && checks.coverage.hasL0 && checks.coverage.hasL1 && new Set(checks.coverage.symbols).size >= 3 && checks.coverage_angle && checks.tracking_global.id === 'fixed_circle_r40_u100' && checks.tracking_global.canvases > 0 && checks.tracking_global.zero && checks.tracking_global.teacherWidth >= 8 && checks.tracking_global.targetDash === 'dash' && checks.tracking_local.id === 'retry19_exact_seam_rectangle_2' && checks.tracking_local.local && checks.tracking_local.projection === 'yz' && checks.tracking_local.zero && checks.tracking_local.teacherMax > 29 && checks.tracking_local.betaNames.join('|') === 'Teacher max |Δβ||L1 raw max |Δβ||7° reference' && checks.tracking_local.betaMax > 29 && checks.tracking_local.rawBetaMax < 1 && checks.toggles.includes('L0 raw Student') && checks.toggles.includes('L0 DLS2') && !checks.toggles.includes('L1 raw Student') && !checks.toggles.includes('L1 DLS2') && checks.theme.dark && checks.theme.trackingGlobal && !checks.theme.overflow && checks.narrow.client === checks.narrow.scroll && exceptions.length === 0 && consoleErrors.length === 0;
    fs.writeFileSync(path.join(root, 'browser_validation.json'), JSON.stringify({
      observed_at: new Date().toISOString(), passed: pass, report_sha256: hash(report), checks, exceptions, consoleErrors,
      engine: 'Google Chrome headless with WebGL SwiftShader',
      boundary: 'HTML rendering and interaction verification only; it is not an artifact rehash or scientific Gate.',
    }, null, 2) + '\n');
    console.log(JSON.stringify({passed: pass, largePathsPassed: checks.largePathsPassed, navigationPassed: checks.navigationPassed, legendPassed: checks.legendPassed, failedLegends: Object.entries(checks.legends).flatMap(([key,rows])=>rows.filter(r=>!r.pass).map(r=>({key,...r}))), exceptions: exceptions.length, consoleErrors: consoleErrors.length}));
    if (!pass) process.exitCode = 1;
  } finally {
    if (ws) ws.close();
    chrome.kill('SIGTERM');
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });

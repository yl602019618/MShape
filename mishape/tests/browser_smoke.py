"""Real Chrome integration checks for MiShape's complete user workflow."""
from pathlib import Path
import json
import time
import zipfile
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'verification'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    checks, errors, timings = [], [], {}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless=True,
            args=['--enable-unsafe-swiftshader', '--use-angle=swiftshader'])
        page = browser.new_page(viewport={'width':1512,'height':982}, device_scale_factor=1, accept_downloads=True)
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto('http://127.0.0.1:8010')
        page.wait_for_function('window.miShape?.model && document.getElementById("loading").classList.contains("hidden")',timeout=60000)
        assert page.evaluate('miShape.model.metadata.asset_id')=='porsche-930'
        assert page.evaluate('miShape.cage.points.length')==84
        checks.append('Two real asset cards, 930 loaded, 84 point cage')
        page.locator('[data-asset="porsche-carrera-4s"]').click()
        page.wait_for_function('miShape.model.metadata.asset_id==="porsche-carrera-4s"',timeout=60000)
        page.wait_for_function('document.getElementById("loading").classList.contains("hidden")')
        baseline = page.evaluate('miShape.analysis.measurements.total_length.value')
        started = time.time()
        page.locator('[data-number="total_length"]').fill('200')
        page.locator('[data-number="total_length"]').press('Tab')
        page.wait_for_function('(v)=>Math.abs(miShape.analysis.measurements.total_length.value-v-200)<.1',arg=baseline,timeout=30000)
        timings['slider_end_to_end_seconds']=round(time.time()-started,3)
        assert page.evaluate('miShape.quality.wheel_rigidity.verified')
        checks.append('Length slider changes actual mesh dimensions, rigid wheel check true')
        page.locator('#undo').click()
        page.wait_for_function('(v)=>Math.abs(miShape.analysis.measurements.total_length.value-v)<.1',arg=baseline)
        page.locator('#redo').click()
        page.wait_for_function('(v)=>Math.abs(miShape.analysis.measurements.total_length.value-v-200)<.1',arg=baseline)
        checks.append('Small-step undo and redo restore exact dimensions')
        page.locator('#compare').click()
        page.wait_for_timeout(150)
        assert page.locator('#baseline-panel').is_visible()
        page.locator('[data-view="side"]').click()
        assert page.evaluate('miViewer.gl.getError()')==0
        page.screenshot(path=str(OUT/'comparison.png'))
        checks.append('Original/design side-by-side with camera presets')
        page.locator('#compare').click()
        page.locator('[data-view="iso"]').click()
        page.locator('[data-tab="cage"]').click()
        page.locator('#control-select').select_option('15')
        page.locator('#control-z').fill('35')
        page.locator('#control-z').press('Tab')
        page.wait_for_function('miShape.controls.some(c=>c.id===15&&Math.abs(c.delta[2]-.035)<1e-8)')
        page.wait_for_function('!document.getElementById("live-indicator").classList.contains("busy")')
        checks.append('Numeric cage point edits trigger mesh deformation')
        # Direct pointer drag uses the same user-accessible Shift modifier.
        xy=page.evaluate('''()=>{let p=miViewer.cagePoints().find(p=>p.index[0]===0&&p.index[1]===0&&p.index[2]===3);let s=miViewer.project(p.position),r=miViewer.canvas.getBoundingClientRect();return [s[0]+r.left,s[1]+r.top]}''')
        page.keyboard.down('Shift');page.mouse.move(*xy);page.mouse.down();page.mouse.move(xy[0]+20,xy[1]-15,steps=4);page.mouse.up();page.keyboard.up('Shift')
        page.wait_for_function('miShape.controls.length>=2')
        page.wait_for_function('!document.getElementById("live-indicator").classList.contains("busy")')
        checks.append('Shift drag of a projected cage handle edits control recipe')
        page.locator('[data-display="wire"]').click();assert page.evaluate('miViewer.wire')
        page.locator('[data-display="heat"]').click();assert page.evaluate('miViewer.heat')
        page.locator('[data-display="studio"]').click()
        page.locator('.part-row').first.click();page.locator('#isolate').check()
        assert page.evaluate('miViewer.isolate && miViewer.selected.size===1')
        page.locator('#parts-all').click()
        page.locator('#explode').evaluate('el=>{el.value=.3;el.dispatchEvent(new Event("input"))}')
        assert page.evaluate('miViewer.explode')==.3
        page.locator('#explode').evaluate('el=>{el.value=0;el.dispatchEvent(new Event("input"))}')
        checks.append('Wire/displacement, component selection, isolation and exploded view')
        saved=page.evaluate('JSON.stringify({parameters:miShape.parameters,controls:miShape.controls,options:miShape.options})')
        with page.expect_download() as downloaded:page.locator('#save').click()
        project=OUT/'replay.mishape.json';downloaded.value.save_as(project)
        assert json.loads(project.read_text())['schema']=='mishape-project-v1'
        page.locator('#reset').click()
        page.wait_for_function('!document.getElementById("live-indicator").classList.contains("busy")')
        old_mid=page.evaluate('miShape.mid')
        page.locator('#import-open').click();page.locator('#import-file').set_input_files(project)
        page.locator('#import-form button[type="submit"]').click()
        page.wait_for_function('(id)=>miShape.mid!==id',arg=old_mid,timeout=60000)
        page.wait_for_function('!document.getElementById("live-indicator").classList.contains("busy")')
        assert page.evaluate('JSON.stringify({parameters:miShape.parameters,controls:miShape.controls,options:miShape.options})')==saved
        checks.append('Project JSON download/import replays parameters and control edits')
        # Calibrate current shaped geometry; a new baseline retains the current form.
        page.locator('[data-tab="measure"]').click()
        page.locator('#known-length').fill('4.6');page.locator('#scale-apply').click()
        page.wait_for_function('miShape.analysis.scale_calibration.status==="user_calibrated"',timeout=60000)
        assert abs(page.evaluate('miShape.analysis.measurements.total_length.value')-4600)<.001
        checks.append('Known-length scale calibration establishes a measured new baseline')
        # Work from the untouched Carrera asset for the portfolio deliverable.
        page.locator('[data-asset="porsche-carrera-4s"]').click()
        page.wait_for_function('miShape.model.name==="Porsche 911 · Carrera 4S" && document.getElementById("loading").classList.contains("hidden")',timeout=60000)
        page.locator('#batch-open').click();page.locator('#batch-count').fill('12');page.locator('#batch-strength').fill('25')
        page.locator('#batch-start').click()
        page.wait_for_function('miShape.batch?.status==="complete"',timeout=180000)
        page.wait_for_function('miShape.variantCache.size===12',timeout=120000)
        assert page.locator('.portfolio-card').count()==12
        page.screenshot(path=str(OUT/'portfolio.png'))
        with page.expect_download() as downloaded:page.locator('#portfolio-download').click()
        batch_path=OUT/'MiShape-Carrera-12-Variants.zip';downloaded.value.save_as(batch_path)
        with zipfile.ZipFile(batch_path) as z:
            assert z.testzip() is None
            manifest=json.loads(z.read('manifest.json'))
            assert manifest['count']==12
            assert len([n for n in z.namelist() if n.endswith('vehicle.obj')])==12
        checks.append('12 actual variants, thumbnail portfolio, downloadable OBJ+recipe ZIP verified')
        expected=page.evaluate('miShape.batch.items[0].recipe.parameters')
        page.locator('[data-variant="0"]').click()
        page.wait_for_function('!document.getElementById("live-indicator").classList.contains("busy")')
        assert page.evaluate('miShape.parameters')==expected
        checks.append('Apply portfolio variant for continued editing')
        page.locator('[data-mode="generate"]').click()
        page.wait_for_function('miShape.lastGen && document.getElementById("loading").classList.contains("hidden")',timeout=60000)
        page.locator('#body-style').select_option('suv')
        page.wait_for_function('miShape.lastGen?.body_style==="suv" && document.getElementById("loading").classList.contains("hidden")',timeout=60000)
        page.locator('[data-number="length"]').fill('5.1');page.locator('[data-number="length"]').press('Tab')
        page.wait_for_function('Math.abs(miShape.analysis.measurements.total_length.value-5100)<.1',timeout=60000)
        checks.append('Parametric generator switches body style and regenerates actual 5.1m vehicle')
        page.screenshot(path=str(OUT/'generator.png'))
        page.locator('#shape-generated').click()
        assert page.evaluate('miShape.mode')=='shape'
        with page.expect_download() as downloaded:
            page.locator('#export-open').click();page.locator('[data-export="glb"]').click()
        generated_path=OUT/'MiShape-Generated-SUV.glb';downloaded.value.save_as(generated_path)
        assert generated_path.read_bytes()[:4]==b'glTF'
        checks.append('Generated vehicle transfers to cage editor and exports valid GLB')
        # Final landing capture: actual Carrera with the editable cage.
        page.locator('[data-asset="porsche-carrera-4s"]').click()
        page.wait_for_function('miShape.model.metadata.asset_id==="porsche-carrera-4s" && document.getElementById("loading").classList.contains("hidden")',timeout=60000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT/'studio.png'))
        assert not errors,errors
        assert page.evaluate('miViewer.gl.getError()')==0
        report={'checks':checks,'count':len(checks),'page_errors':errors,'webgl_error':0,'timings':timings,
                'browser':'Local Chrome, headless, software WebGL2','batch_id':manifest['model_id'],'batch_count':12}
        (OUT/'browser-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False,indent=2))
        browser.close()


if __name__=='__main__':main()

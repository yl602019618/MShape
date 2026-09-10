"""Chrome checks for fitted cages, density changes and legacy project replay."""
import json
import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / 'docs' / 'verification'
URL = os.environ.get('MISHAPE_TEST_URL', 'http://127.0.0.1:8010')
RECIPE = '({parameters:miShape.parameters,controls:miShape.controls,options:miShape.options})'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    errors, checks, reports = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless=True,
            args=['--enable-unsafe-swiftshader', '--use-angle=swiftshader'])
        page = browser.new_page(viewport={'width': 1512, 'height': 982}, device_scale_factor=1, accept_downloads=True)
        page.on('pageerror', lambda e: errors.append(str(e)))

        def idle():
            page.wait_for_function('window.miShape?.model && document.getElementById("loading").classList.contains("hidden") && !document.getElementById("live-indicator").classList.contains("busy")', timeout=90000)

        def apply(preset, dims):
            page.locator('#cage-density').select_option(preset)
            page.locator('#cage-apply').click()
            page.wait_for_function('(dims)=>miShape.cage.dimensions.join()===dims.join() && document.getElementById("loading").classList.contains("hidden")', arg=dims, timeout=90000)
            idle()

        page.goto(URL)
        idle()
        for asset in ['porsche-930', 'porsche-carrera-4s']:
            page.locator(f'[data-asset="{asset}"]').click()
            page.wait_for_function('(asset)=>miShape.model.metadata.asset_id===asset', arg=asset, timeout=60000)
            idle()
            page.locator('[data-tab="cage"]').click()
            assert page.evaluate('miShape.cage.type') == 'fitted'
            assert page.evaluate('miShape.cage.points.length') == 108
            assert page.evaluate('miViewer.cagePoints().length') == 94
            assert page.locator('#control-select option').count() == 94
            profile = page.evaluate('miShape.cage.points.filter(p=>p.index[1]===1&&p.index[2]===3).map(p=>p.rest[2])')
            assert max(profile) - profile[0] > .4
            assert max(profile) - profile[-1] > .3
            checks.append(asset + ': fitted roof/hood/tail profile and boundary-only picking')

            page.locator('[data-tab="parameters"]').click()
            page.locator('[data-number="total_length"]').fill('120')
            page.locator('[data-number="total_length"]').press('Tab')
            idle()
            page.evaluate('window.beforeDensity = Array.from(miViewer.vertices)')
            page.locator('[data-tab="cage"]').click()
            apply('coarse', [5, 3, 3])
            assert page.evaluate('miShape.lastRegrid.max_error_mm') == 0
            assert page.evaluate('miViewer.vertices.every((v,i)=>v===beforeDensity[i])')
            apply('standard', [9, 3, 4])
            assert page.evaluate('miViewer.vertices.every((v,i)=>v===beforeDensity[i])')
            checks.append(asset + ': pure parameter geometry identical across densities')

            # A real local roof edit, then refinement and reversible coarsening.
            page.locator('#control-select').select_option('55')
            page.locator('#control-z').fill('60')
            page.locator('#control-z').press('Tab')
            idle()
            assert page.evaluate('miShape.controls.some(c=>c.id===55&&Math.abs(c.delta[2]-.06)<1e-9)')
            before = page.evaluate(RECIPE)
            page.evaluate('window.beforeDensity = Array.from(miViewer.vertices)')
            started = time.time()
            apply('fine', [13, 5, 5])
            transfer = page.evaluate('miShape.lastRegrid')
            transfer['asset'] = asset
            transfer['browser_seconds'] = round(time.time() - started, 3)
            reports.append(transfer)
            assert transfer['max_error_mm'] < 4
            assert transfer['rms_error_mm'] < .4
            assert page.locator('#control-select option').count() == 226
            assert page.evaluate('miShape.controls.length') > 1
            assert page.evaluate('miShape.parameters.total_length') == 120
            assert page.locator('#cage-regrid-report').is_visible()
            measured = page.evaluate('''()=>{let max=0;for(let i=0;i<beforeDensity.length;i+=3)max=Math.max(max,Math.hypot(miViewer.vertices[i]-beforeDensity[i],miViewer.vertices[i+1]-beforeDensity[i+1],miViewer.vertices[i+2]-beforeDensity[i+2])*1000);return max}''')
            assert abs(measured - transfer['max_error_mm']) < .001
            assert page.evaluate('miShape.quality.topology_preserved && miShape.quality.wheel_rigidity.verified')
            page.locator('#undo').click()
            idle()
            assert page.evaluate(RECIPE) == before
            assert page.evaluate('miViewer.vertices.every((v,i)=>v===beforeDensity[i])')
            assert page.locator('#cage-regrid-report').is_hidden()
            page.locator('#redo').click()
            idle()
            assert page.evaluate('miShape.cage.dimensions') == [13, 5, 5]
            apply('coarse', [5, 3, 3])
            coarse = page.evaluate('miShape.lastRegrid')
            assert coarse['max_error_mm'] > 2
            assert 'review' in page.locator('#cage-regrid-report').get_attribute('class')
            page.locator('[data-view="side"]').click()
            page.screenshot(path=str(OUT / f'cage-{asset}-coarse.png'))
            page.locator('#undo').click()
            idle()
            checks.append(asset + ': manual displacement transferred; measured error, warning, undo/redo and rigid wheels verified')

        # Custom dimensions and padding form an exportable, replayable recipe.
        page.locator('#cage-density').select_option('custom')
        page.locator('#cage-nx').fill('11')
        page.locator('#cage-ny').select_option('5')
        page.locator('#cage-nz').fill('4')
        page.locator('#cage-padding').fill('45')
        page.locator('#cage-apply').click()
        page.wait_for_function('miShape.cage.dimensions.join() === "11,5,4"', timeout=90000)
        idle()
        assert page.evaluate('miShape.options.cage.padding_mm') == 45
        assert page.evaluate('miShape.cage.points.length') == 220
        custom_recipe = page.evaluate(RECIPE)
        page.evaluate('window.customVertices=Array.from(miViewer.vertices)')
        with page.expect_download() as download:
            page.locator('#save').click()
        saved = OUT / 'cage-replay.mishape.json'
        download.value.save_as(saved)
        document = json.loads(saved.read_text())
        assert document['recipe'] == custom_recipe
        mid = page.evaluate('miShape.mid')
        page.locator('#import-open').click()
        page.locator('#import-file').set_input_files(saved)
        page.locator('#import-form button[type="submit"]').click()
        page.wait_for_function('(mid)=>miShape.mid!==mid', arg=mid, timeout=90000)
        idle()
        assert page.evaluate(RECIPE) == custom_recipe
        assert page.evaluate('miViewer.vertices.every((v,i)=>v===customVertices[i])')
        checks.append('Custom 11×5×4 / 45 mm settings and transferred controls survive JSON roundtrip exactly')

        # An actual pre-0.2 project omits cage entirely. The UI must restore zero padding.
        document['recipe'] = {'parameters': {'total_length': 80}, 'controls': [{'id': 35, 'delta': [0, 0, .01]}], 'options': {'symmetry': True, 'preserve_wheels': True}}
        old = OUT / 'cage-legacy.mishape.json'
        old.write_text(json.dumps(document))
        mid = page.evaluate('miShape.mid')
        expected = page.request.post(URL + f'/api/models/{mid}/deform', data=document['recipe']).json()['vertices']
        page.locator('#import-open').click()
        page.locator('#import-file').set_input_files(old)
        page.locator('#import-form button[type="submit"]').click()
        page.wait_for_function('(mid)=>miShape.mid!==mid', arg=mid, timeout=90000)
        idle()
        assert page.evaluate('miShape.options.cage') == {'type': 'box', 'dimensions': [7, 3, 4], 'padding_mm': 0}
        assert page.evaluate('(expected)=>miViewer.vertices.every((v,i)=>v===expected[i])', expected)
        checks.append('Old project restores original box84/zero padding and exact legacy geometry')

        # Final presentation is the untouched car, fitted standard cage.
        page.locator('[data-asset="porsche-carrera-4s"]').click()
        idle()
        page.locator('[data-tab="cage"]').click()
        page.locator('[data-view="iso"]').click()
        page.screenshot(path=str(OUT / 'cage-fitted-studio.png'))
        page.locator('[data-view="side"]').click()
        page.screenshot(path=str(OUT / 'cage-fitted-side.png'))
        apply('fine', [13, 5, 5])
        page.locator('[data-view="iso"]').click()
        page.screenshot(path=str(OUT / 'cage-fitted-fine.png'))
        assert not errors, errors
        assert page.evaluate('miViewer.gl.getError()') == 0
        report = {'checks': checks, 'count': len(checks), 'transfers': reports, 'page_errors': errors, 'webgl_error': 0}
        (OUT / 'cage-browser-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        browser.close()


if __name__ == '__main__':
    main()

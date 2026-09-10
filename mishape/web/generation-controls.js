/* Physical slider reconciliation for MiShape's v1 polygon generator.
 * All lengths are metres, angles degrees. This is a deterministic layout check,
 * not a substitute for the backend's full surface-quality screening.
 */
const EPS = 1e-9;
const tan = angle => Math.tan(angle * Math.PI / 180);
const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
const label = (fields, key) => fields.get(key)?.label || key;

function fieldsOf(schema) {
  const list = Array.isArray(schema) ? schema : schema?.fields;
  if (!Array.isArray(list)) throw Error('生成参数 schema 必须提供 fields 数组。');
  return new Map(list.map(field => [field.id || field.key, field]));
}

function scalarBounds(fields, key) {
  const field = fields.get(key);
  if (!field || !Number.isFinite(field.min) || !Number.isFinite(field.max)) throw Error(`缺少 ${key} 的物理参数范围。`);
  return [field.min, field.max];
}

function layout(p) {
  const overhang = p.length - p.wheelbase;
  const front = -p.wheelbase / 2;
  const minX = front - overhang * p.front_overhang_ratio;
  const maxX = p.wheelbase / 2 + overhang * (1 - p.front_overhang_ratio);
  const roofFront = p.height - p.roof_crown;
  const windRun = (roofFront - p.hood_height) / tan(p.windscreen_angle);
  const rearRun = (roofFront - p.roof_rear_drop - p.decklid_height) / tan(p.backlight_angle);
  const gateRun = (p.decklid_height - p.tailgate_lower_height) / tan(p.tailgate_angle);
  const cabinAnchor = maxX - gateRun - p.decklid_length - rearRun - windRun;
  return {front, minX, maxX, windRun, rearRun, gateRun, cabinAnchor};
}

/**
 * Reconcile one user edit without mutating either argument.
 *
 * `schema` accepts generation_schema() or generation_schema().fields.
 * `values` should be the selected body's complete preset/current slider state.
 * Missing controls fall back to schema defaults. The user's edited field is
 * never overwritten. `changes` contains ONLY automatic linked changes.
 * On rejection, return the original values, an empty changes list and a readable
 * error. Call this function only when the UI's automatic linkage is enabled.
 */
export function reconcileGeneration(values, key, value, schema) {
  const previous = values && typeof values === 'object' && !Array.isArray(values) ? {...values} : {};
  const reject = message => ({values: {...previous}, changes: [], error: message});
  try {
    if (!values || typeof values !== 'object' || Array.isArray(values)) return reject('生成参数必须是对象。');
    const fields = fieldsOf(schema), field = fields.get(key);
    if (!field) return reject(`未知生成参数：${key}。`);
    const p = {...previous};
    for (const [id, specification] of fields) {
      if (p[id] === undefined && specification.default !== undefined) p[id] = specification.default;
    }
    if (field.type === 'select') {
      const allowed = (field.options || []).map(option => typeof option === 'object' ? option.value : option);
      if (!allowed.includes(value)) return reject(`${label(fields, key)} 选项无效。`);
    } else {
      if (typeof value !== 'number' || !Number.isFinite(value)) return reject(`${label(fields, key)} 必须是有限数值。`);
      if (value < field.min - EPS || value > field.max + EPS) return reject(`${label(fields, key)} 范围为 ${field.min}–${field.max} ${field.unit || ''}。`);
    }
    p[key] = value;
    // Invalid saved state is rejected explicitly, rather than propagated to NaN.
    for (const [id, specification] of fields) {
      if (specification.type === 'select') continue;
      if (typeof p[id] !== 'number' || !Number.isFinite(p[id]) || p[id] < specification.min - EPS || p[id] > specification.max + EPS) {
        return reject(`${label(fields, id)} 的当前值超出 ${specification.min}–${specification.max}；请恢复构型预设或调整该参数。`);
      }
    }
    const changes = new Map();
    const assign = (id, target) => {
      if (id === key) throw Error(`无法在保留 ${label(fields, key)} 输入值时满足当前布置。`);
      const [minimum, maximum] = scalarBounds(fields, id);
      if (!Number.isFinite(target) || target < minimum - EPS || target > maximum + EPS) throw Error(`${label(fields, id)} 无法在 ${minimum}–${maximum} 的范围内完成联动。`);
      target = clamp(target, minimum, maximum);
      if (Math.abs(target - p[id]) <= EPS) return;
      const from = changes.get(id)?.from ?? p[id];
      p[id] = target;
      changes.set(id, {key: id, from, to: target});
    };

    // Keep wheelbase fixed when editing length. Adjust only overhang balance if
    // needed; if neither axle nor length can remain fixed, reject the edit.
    const overhang = p.length - p.wheelbase;
    const [ratioMinimum, ratioMaximum] = scalarBounds(fields, 'front_overhang_ratio');
    if (overhang <= 0) return reject('整车长度必须大于轴距，并为前悬和后悬留下空间。');
    const ratioLow = Math.max(ratioMinimum, .68 / overhang, 1 - 1.18 / overhang);
    const ratioHigh = Math.min(ratioMaximum, 1.12 / overhang, 1 - .55 / overhang);
    if (ratioLow > ratioHigh + EPS) {
      return reject(`当前轴距 ${p.wheelbase.toFixed(3)} m 无法容纳该总长；前后悬合计须为 1.230–2.300 m，请调整总长或轴距。`);
    }
    if (p.front_overhang_ratio < ratioLow - EPS || p.front_overhang_ratio > ratioHigh + EPS) {
      if (key === 'front_overhang_ratio') return reject(`当前总长和轴距下，前悬占比须为 ${ratioLow.toFixed(4)}–${ratioHigh.toFixed(4)}。`);
      const guard = Math.min(1e-6, Math.max(0, (ratioHigh - ratioLow) / 4));
      assign('front_overhang_ratio', clamp(p.front_overhang_ratio, ratioLow + guard, ratioHigh - guard));
    }

    // A width edit controls the body; track/tyre edits control the wheel assembly.
    // No wheel-radius change or implicit tyre-width change is used to hide a fit.
    if (p.track + p.tyre_width > p.width + .10 + EPS) {
      if (key === 'width') {
        const target = Math.floor((p.width + .10 - p.tyre_width + 1e-10) * 1e6) / 1e6;
        assign('track', target);
      } else if (key === 'track' || key === 'tyre_width') {
        const target = Math.ceil((p.track + p.tyre_width - .10 - 1e-10) * 1e6) / 1e6;
        assign('width', target);
      } else {
        return reject('当前轮距与胎宽超过车宽适配域；请调整车宽、轮距或胎宽。');
      }
    }

    // Height remains a user-controlled physical dimension. Do not automatically
    // rewrite the deck, beltline, tailgate, glass angles and hood to force a fit.
    const requiredHeight = Math.max(
      scalarBounds(fields, 'height')[0], p.decklid_height + .22,
      p.roof_crown + p.roof_rear_drop + p.decklid_height + .16,
      p.roof_crown + p.hood_height + .12 * tan(p.windscreen_angle));
    if (p.height < requiredHeight - EPS) {
      return reject(`保留当前车顶、甲板与风挡参数时，车高至少为 ${requiredHeight.toFixed(3)} m；请提高车高或手动调整关联造型。`);
    }
    if (p.decklid_height - p.rear_belt_height < .045 - EPS) return reject('后玻璃下缘须高于后腰线至少 45 mm；请调整甲板或腰线高度。');
    if (p.decklid_height - p.tailgate_lower_height < .10 - EPS) return reject('尾门上下缘高度差须至少 100 mm；请调整甲板或尾门下缘高度。');

    const geometric = layout(p);
    if (!Object.values(geometric).every(Number.isFinite) || geometric.windRun < .12 - EPS || geometric.rearRun <= .001 || geometric.gateRun <= .001) {
      return reject('当前玻璃角度与高度无法形成有效的前后风挡控制段，请调整玻璃角度或高度。');
    }
    const [cabinMinimum, cabinMaximum] = scalarBounds(fields, 'cabin_length');
    const minimum = Math.max(cabinMinimum, geometric.cabinAnchor - (geometric.front + 1.05));
    const maximum = Math.min(cabinMaximum, geometric.cabinAnchor - (geometric.minX + .58));
    if (minimum > maximum + EPS) return reject('该尺寸组合没有可用的座舱布置；请调整总长、轴距、尾箱平台或玻璃角度。');
    if (p.cabin_length < minimum - EPS || p.cabin_length > maximum + EPS) {
      if (key === 'cabin_length') return reject(`当前车身布置下，车顶控制段须为 ${minimum.toFixed(3)}–${maximum.toFixed(3)} m。`);
      // Five millimetres of interior slack avoids repeated boundary rejection.
      // Round the linked length to a visible millimetre when the interval allows.
      const guard = Math.min(.005, Math.max(0, (maximum - minimum) / 4));
      let target = clamp(p.cabin_length, minimum + guard, maximum - guard);
      const rounded = p.cabin_length < minimum ? Math.ceil(target * 1000) / 1000 : Math.floor(target * 1000) / 1000;
      if (rounded >= minimum && rounded <= maximum) target = rounded;
      assign('cabin_length', target);
    }
    return {values: p, changes: [...changes.values()]};
  } catch (error) {
    return reject(error instanceof Error ? error.message : String(error));
  }
}

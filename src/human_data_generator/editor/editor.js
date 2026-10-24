import {mountMap} from '/editor_map.js';

const $ = id => document.getElementById(id);
const clone = value => structuredClone(value);
let labelCounter = 0;
const state = {scene: null, rules: null, active: 0, components: null, inspected: null,
               sceneSaved: null, rulesSaved: null, editRevision: 0, frame: 0};
const map = mountMap($('map'), () => state.inspected || state.scene, () => {sceneEdited(); renderScene();});
const mapHint = '橙点：当前帧人物；青/紫箭头：cam2/cam4 光轴方向';
let baseTrajectory = null;
let trajectoryRequest = 0;
let trajectoryTimer = null;

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
function node(tag, text, className) {
  const value = document.createElement(tag);
  if (text !== undefined) value.textContent = text;
  if (className) value.className = className;
  return value;
}
function label(parent, text, control) {
  const element = node('label', text);
  if (control) {
    if (!control.id) control.id = `field_${++labelCounter}`;
    element.htmlFor = control.id;
  }
  parent.append(element);
  return element;
}
function numberField(parent, title, value, onChange, {step = 'any', min, id} = {}) {
  const input = node('input');
  input.type = 'number'; input.step = step; input.value = value;
  if (min !== undefined) input.min = min;
  if (id) input.id = id;
  label(parent, title, input);
  input.addEventListener('change', () => onChange(Number(input.value)));
  parent.append(input);
  return input;
}
function button(text, onClick) {
  const result = node('button', text);
  result.type = 'button'; result.onclick = onClick;
  return result;
}
function show(message, error = false) {
  const panel = $('feedback');
  panel.replaceChildren();
  panel.classList.toggle('error', error);
  if (!error) {panel.textContent = message; return;}
  const raw = String(message);
  let explanation = /[\u4e00-\u9fff]/.test(raw) ? raw : '当前设置未通过检查。';
  if (raw.includes('invalid_template_id')) explanation = '模板编号只能填写 1–48 个英文小写字母、数字或下划线，例如 city_example。';
  else if (raw.includes('motion_exceeds_cache')) explanation = '源动作剩余时间不够覆盖整段场景。请缩短帧数、把源动作起始帧调早，或选更长的动作。';
  else if (raw.includes('File exists')) explanation = '同名规则文件已经存在，请换一个导出名称。';
  else if (raw.includes('invalid_batch_rules')) explanation = '规则文件格式不受支持，请使用当前编辑器导出的文件。';
  else if (raw.includes('range') || raw.includes('offset_xy')) explanation = '检查范围的最小值、最大值和单位。';
  else if (raw.includes('unknown_') || raw.includes('unavailable_') || raw.includes('candidate_conflict') || raw.includes('no_usable_')) explanation = '组件候选不可用或与筛选条件、身体动作不兼容。';
  else if (/invalid_person_rule:\d+/.test(raw)) explanation = `模板 ${raw.split(':')[0]} 的随机设置引用了人物 ${raw.match(/invalid_person_rule:(\d+)/)[1]}，但该模板的基准场景没有这个人物。模板编号不代表当前环境。`;
  else if (raw.includes('invalid_person_rule')) explanation = '人物随机设置包含不支持的字段。';
  else if (raw.includes('invalid_rig_rule')) explanation = '相机随机设置引用了不存在的相机组，或填写了不支持的字段。';
  else if (raw.includes('outside_verified_road')) explanation = '变体路线离开已核对道路，请缩小路线偏移。';
  const person = raw.match(/person_(\d+)/);
  const rig = raw.match(/rig_[a-z0-9_]+/);
  panel.append(node('div', `${person ? `人物 ${person[1]}：` : rig ? `${rig[0]}：` : ''}${explanation}`));
  const details = node('details');
  details.append(node('summary', '技术详情'), node('pre', raw));
  panel.append(details);
}
function fail(error) { show(error.message || String(error), true); }
function template() { return state.rules.templates[state.active]; }
function updateBadge() {
  const sceneDirty = state.sceneSaved !== JSON.stringify(state.scene);
  const rulesDirty = state.rulesSaved !== JSON.stringify(state.rules);
  $('dirty').textContent = `${sceneDirty ? '场景未保存为修订' : '场景修订已保存'} · ${rulesDirty ? '规则未导出' : '规则已导出'}`;
}
function clearPreview(message = '场景已变化，旧预览不代表当前设置。') {
  for (const camera of ['cam2', 'cam4']) {
    $(camera).removeAttribute('src');
    $(camera + 'Link').removeAttribute('href');
  }
  $('previewNote').textContent = message;
}
function trajectoryProblem(reason) {
  const person = reason.match(/person_(\d+)/)?.[1];
  const frame = reason.match(/frame_(\d+)/)?.[1];
  const label = person ? `人物 ${person}` : '人物';
  if (reason.includes('route_time_out_of_bounds')) {
    return `${label}在第 ${frame} 帧走出了路线，请延长路线或降低速度。`;
  }
  return `${label}的路线无法计算（${reason}）。`;
}
function refreshTrajectory(scene = state.scene) {
  const request = ++trajectoryRequest;
  clearTimeout(trajectoryTimer);
  if (scene === state.scene) baseTrajectory = null;
  map.setTrajectory(null);
  $('timelineNote').textContent = '正在计算当前帧的人物位置…';
  trajectoryTimer = setTimeout(async () => {
    try {
      const positions = await api('/api/trajectory', {scene: clone(scene)});
      if (request !== trajectoryRequest) return;
      if (scene === state.scene) baseTrajectory = positions;
      if (scene === (state.inspected || state.scene)) {
        map.setTrajectory(positions);
        const errors = Object.values(positions.errors || {});
        const missing = scene.people
          .filter(person => !positions.people?.[String(person.person_id)] && !positions.errors?.[String(person.person_id)])
          .map(person => person.person_id);
        if (errors.length) {
          const detail = errors.map(trajectoryProblem).join(' ');
          $('mapHint').textContent = detail;
          $('timelineNote').textContent = `${detail} 该人物此前有效帧及其他人物的位置仍显示。`;
        } else if (missing.length) {
          const detail = `人物 ${missing.join('、')} 没有返回位置；请重新计算当前场景。`;
          $('mapHint').textContent = detail;
          $('timelineNote').textContent = detail;
        } else {
          $('mapHint').textContent = state.inspected ? '正在查看试抽结果；橙色圆点显示当前帧路线位置' : mapHint;
          $('timelineNote').textContent = `已计算 ${scene.people.length} 个人物；地图上的橙色数字圆点显示当前帧位置。`;
        }
      }
    } catch (error) {
      if (request === trajectoryRequest) {
        map.setTrajectory(null);
        const reason = error.message === 'unknown_route'
          ? '当前编辑器服务仍是旧后端。请在空闲端口启动新版服务，才能显示随时间移动的人物。'
          : `人物位置暂不可显示：${error.message}`;
        $('mapHint').textContent = reason;
        $('timelineNote').textContent = reason;
      }
    }
  }, 80);
}
function sceneEdited() {
  state.editRevision += 1;
  state.inspected = null;
  $('backToBase').hidden = true;
  map.setReadonly(false);
  $('mapHint').textContent = mapHint;
  clearPreview();
  $('samples').replaceChildren();
  updateBadge();
  refreshTrajectory();
}
function rulesEdited() { $('samples').replaceChildren(); updateBadge(); }
function getReady(kind) { return state.components[kind].filter(item => item.status === 'ready'); }
function selectField(parent, title, value, rows, onChange) {
  const control = node('select');
  for (const row of rows) control.add(new Option(row.label || row.id, row.id));
  control.value = value;
  label(parent, title, control);
  control.onchange = () => onChange(control.value);
  parent.append(control);
  return control;
}
function renderScene() {
  const scene = state.scene;
  $('environment').value = scene.environment;
  $('sceneId').value = scene.scene_id;
  $('sceneSeed').value = scene.seed;
  $('fps').value = scene.fps;
  $('frameCount').value = scene.frame_count;
  state.frame = Math.max(0, Math.min(state.frame, scene.frame_count - 1));
  $('frame').max = scene.frame_count - 1;
  $('frame').value = state.frame;
  $('frameLabel').textContent = `${state.frame} · ${(state.frame / scene.fps).toFixed(2)} 秒`;
  map.setFrame(state.frame);
  $('people').replaceChildren();
  for (const person of scene.people) {
    const card = node('div', undefined, 'card');
    const heading = node('h3', `人物 ${person.person_id}`);
    card.append(heading, button('移除人物', () => {
      if (scene.people.length === 1) return show('至少保留一个人物。', true);
      scene.people = scene.people.filter(p => p.person_id !== person.person_id);
      template().people = (template().people || []).filter(p => p.person_id !== person.person_id);
      sceneEdited(); renderScene(); renderRules();
    }));
    for (const [field, kind, title] of [
      ['body', 'bodies', '身体'], ['motion', 'motions', '动作'],
      ['clothing', 'clothing', '衣物'], ['hair', 'hair', '发型'],
    ]) {
      selectField(card, title, person[field], getReady(kind), value => {
        person[field] = value;
        sceneEdited(); renderScene(); renderRules();
      });
    }
    const motion = state.components.motions.find(item => item.id === person.motion);
    const clothing = state.components.clothing.find(item => item.id === person.clothing);
    const note = node('p', motion ? `动作可用 ${motion.available_frames} 帧；起始帧还须给整段场景留够长度。` : '', 'hint');
    if (clothing?.kind === 'native_locked') note.textContent += ` ${person.clothing} 仅适用于 ${clothing.locked_body} + ${clothing.locked_motion}、源帧 ${clothing.locked_window.join('–')}。`;
    card.append(note);
    numberField(card, '动作起始帧（源动作中的位置）', person.motion_start, value => {person.motion_start = value; sceneEdited();}, {step: 1, min: 0});
    numberField(card, '沿路线速度（米/秒）', person.speed_m_s, value => {person.speed_m_s = value; sceneEdited();}, {min: 0});
    numberField(card, '路线起始进度（米）', person.start_progress_m, value => {person.start_progress_m = value; sceneEdited();}, {min: 0});
    const route = node('textarea'); route.rows = Math.max(3, person.route_xy_m.length);
    route.value = person.route_xy_m.map(point => point.join(', ')).join('\n');
    label(card, '路线控制点 X, Y（每行一个点，米）', route);
    route.onchange = () => {
      const points = route.value.trim().split('\n').map(line => line.split(',').map(Number));
      if (points.length < 2 || points.some(point => point.length !== 2 || point.some(v => !Number.isFinite(v)))) return show(`人物 ${person.person_id} 的路线需要至少两个有效控制点。`, true);
      person.route_xy_m = points; sceneEdited(); map.draw();
    };
    card.append(route);
    $('people').append(card);
  }
  $('rigs').replaceChildren();
  for (const rig of scene.rigs) {
    const card = node('div', undefined, 'card');
    card.append(node('h3', rig.rig_id), button('移除相机组', () => {
      if (scene.rigs.length === 1) return show('至少保留一个相机组。', true);
      scene.rigs = scene.rigs.filter(r => r.rig_id !== rig.rig_id);
      template().rigs = (template().rigs || []).filter(r => r.rig_id !== rig.rig_id);
      sceneEdited(); renderScene(); renderRules();
    }));
    const row = node('div', undefined, 'fields3');
    for (const [title, value, apply] of [
      ['X（米）', rig.origin_m[0], v => rig.origin_m[0] = v],
      ['Y（米）', rig.origin_m[1], v => rig.origin_m[1] = v],
      ['朝向（度）', rig.yaw_deg, v => rig.yaw_deg = v],
    ]) {
      const cell = node('div'); numberField(cell, title, value, v => {apply(v); sceneEdited();}, {}); row.append(cell);
    }
    card.append(row); $('rigs').append(card);
  }
  const selected = $('previewRig').value;
  $('previewRig').replaceChildren(...scene.rigs.map(r => new Option(r.rig_id, r.rig_id)));
  if (scene.rigs.some(r => r.rig_id === selected)) $('previewRig').value = selected;
  map.draw(); updateBadge();
}

function personSpec(pid) {
  const rule = template();
  rule.people ||= [];
  let spec = rule.people.find(p => p.person_id === pid);
  if (!spec) {spec = {person_id: pid}; rule.people.push(spec);}
  return spec;
}
function changePerson(pid, key, value) {
  const spec = personSpec(pid);
  if (value === null) delete spec[key]; else spec[key] = value;
  if (Object.keys(spec).length === 1) template().people = template().people.filter(p => p !== spec);
  rulesEdited();
}
function changeRig(rid, key, value) {
  const rule = template(); rule.rigs ||= [];
  let spec = rule.rigs.find(r => r.rig_id === rid);
  if (!spec) {spec = {rig_id: rid}; rule.rigs.push(spec);}
  if (value === null) delete spec[key]; else spec[key] = value;
  if (Object.keys(spec).length === 1) rule.rigs = rule.rigs.filter(r => r !== spec);
  rulesEdited();
}
function toggle(parent, title, checked, onChange) {
  const row = node('label');
  const input = node('input'); input.type = 'checkbox'; input.checked = checked;
  input.onchange = () => onChange(input.checked);
  row.append(input, document.createTextNode(title)); parent.append(row);
  return input;
}
function rangeEditor(parent, title, value, fixed, apply, {integer = false, unit = ''} = {}) {
  const box = node('div', undefined, 'card');
  toggle(box, `${title}：随机变化 ${unit}`, value !== undefined, checked => {
    apply(checked ? [fixed, fixed] : null); renderRules();
  });
  if (value !== undefined) {
    const row = node('div', undefined, 'fields2');
    const latest = [...value];
    for (const [i, name] of [[0, '最小'], [1, '最大']]) {
      const cell = node('div');
      numberField(cell, name, latest[i], next => {latest[i] = next; apply([...latest]);}, {step: integer ? 1 : 'any'});
      row.append(cell);
    }
    box.append(row);
  } else box.append(node('p', `固定：${fixed} ${unit}`, 'hint'));
  parent.append(box);
}
function xyEditor(parent, title, value, apply) {
  const box = node('div', undefined, 'card');
  toggle(box, `${title}：随机变化（相对基准，米）`, value !== undefined, checked => {
    apply(checked ? [[0, 0], [0, 0]] : null); renderRules();
  });
  if (value !== undefined) {
    const grid = node('div', undefined, 'fields2');
    const latest = clone(value);
    for (const [axis, name] of [[0, 'X'], [1, 'Y']]) {
      const group = node('div');
      for (const [end, edge] of [[0, '最小'], [1, '最大']]) {
        numberField(group, `${name} ${edge}`, latest[axis][end], next => {
          latest[axis][end] = next; apply(clone(latest));
        });
      }
      grid.append(group);
    }
    box.append(grid);
  }
  parent.append(box);
}
function componentPolicy(kind, person) {
  if (kind === 'body') return {default_versions_only: true, ids: []};
  if (kind === 'motion') {
    const row = state.components.motions.find(item => item.id === person.motion);
    return {behavior: row?.behavior || 'walk', ids: []};
  }
  if (kind === 'clothing') return {kinds: ['simple_barycentric'], ids: []};
  return {ids: []};
}
function componentEditor(parent, person, kind, title, policy) {
  const box = node('div', undefined, 'card');
  toggle(box, `${title}：随机候选`, policy !== undefined, checked => {
    const spec = personSpec(person.person_id);
    spec.components ||= {};
    if (checked) spec.components[kind] = componentPolicy(kind, person);
    else delete spec.components[kind];
    if (!Object.keys(spec.components).length) delete spec.components;
    if (Object.keys(spec).length === 1) template().people = template().people.filter(p => p !== spec);
    rulesEdited(); renderRules();
  });
  if (policy === undefined) {
    box.append(node('p', `固定为基准场景的 ${person[kind]}`, 'hint'));
    parent.append(box); return;
  }
  const apply = () => {rulesEdited();};
  if (kind === 'body') toggle(box, '只用默认版本', policy.default_versions_only, checked => {policy.default_versions_only = checked; apply();});
  if (kind === 'motion') selectField(box, '动作类别', policy.behavior, [
    {id: 'walk', label: '步行'}, {id: 'ride', label: '骑行'},
  ], value => {policy.behavior = value; apply();});
  if (kind === 'clothing') {
    for (const [id, text] of [['simple_barycentric', '可重绑衣物'], ['native_locked', '限定身体/动作/源帧的衣物']]) {
      toggle(box, text, policy.kinds.includes(id), checked => {
        policy.kinds = checked ? [...policy.kinds, id] : policy.kinds.filter(x => x !== id);
        apply();
      });
    }
  }
  const details = node('details');
  details.append(node('summary', '限定具体候选（不勾选＝筛选后全部）'));
  const list = node('div', undefined, 'candidates');
  for (const item of getReady({body: 'bodies', motion: 'motions', clothing: 'clothing', hair: 'hair'}[kind])) {
    toggle(list, item.id + (item.kind === 'native_locked' ? ` · 仅 ${item.locked_body} / ${item.locked_motion} / ${item.locked_window.join('–')}` : ''), policy.ids.includes(item.id), checked => {
      policy.ids = checked ? [...policy.ids, item.id] : policy.ids.filter(id => id !== item.id);
      apply();
    });
  }
  details.append(list); box.append(details); parent.append(box);
}
function renderRules() {
  const current = template();
  const selected = state.active;
  const environmentName = {city: '城市', riverfront: '河岸'};
  $('templateSelect').replaceChildren(...state.rules.templates.map((t, index) =>
    new Option(`${t.template_id} · ${environmentName[t.base_scene.environment] || t.base_scene.environment}`, String(index))));
  $('templateSelect').value = String(selected);
  $('templateId').value = current.template_id;
  $('batchSeed').value = state.rules.seed;
  $('sharedOffset').replaceChildren();
  xyEditor($('sharedOffset'), '所有人物路线一起平移；相对关系不变', current.route_offset_xy_m,
    value => {if (value === null) delete current.route_offset_xy_m; else current.route_offset_xy_m = value; rulesEdited();});
  $('personRules').replaceChildren();
  for (const person of state.scene.people) {
    const spec = current.people?.find(p => p.person_id === person.person_id) || {};
    const card = node('div', undefined, 'card'); card.append(node('h3', `人物 ${person.person_id}`));
    for (const [kind, title] of [['body', '身体'], ['motion', '动作'], ['clothing', '衣物'], ['hair', '发型']]) {
      componentEditor(card, person, kind, title, spec.components?.[kind]);
    }
    for (const [key, title, unit] of [
      ['speed_m_s', '速度', '米/秒'], ['motion_start', '源动作起始帧', '帧'],
      ['start_progress_m', '路线起始进度', '米（沿整条路线已前进的距离）'],
    ]) {
      rangeEditor(card, title, spec[key], person[key], value => changePerson(person.person_id, key, value),
                  {integer: key === 'motion_start', unit});
    }
    xyEditor(card, '个人路线整体平移；可能改变人与人的距离', spec.route_offset_xy_m,
      value => changePerson(person.person_id, 'route_offset_xy_m', value));
    $('personRules').append(card);
  }
  $('rigRules').replaceChildren();
  for (const rig of state.scene.rigs) {
    const spec = current.rigs?.find(r => r.rig_id === rig.rig_id) || {};
    const card = node('div', undefined, 'card'); card.append(node('h3', rig.rig_id));
    xyEditor(card, '相机组整体 XY 平移', spec.offset_xy_m,
      value => changeRig(rig.rig_id, 'offset_xy_m', value));
    rangeEditor(card, '相机组朝向偏移', spec.offset_yaw_deg, 0,
      value => changeRig(rig.rig_id, 'offset_yaw_deg', value), {unit: '度，相对基准'});
    $('rigRules').append(card);
  }
  updateBadge();
}

function reconcileTargets(rule, scene) {
  const people = new Set(scene.people.map(person => person.person_id));
  const rigs = new Set(scene.rigs.map(rig => rig.rig_id));
  const removedPeople = (rule.people || []).filter(person => !people.has(person.person_id))
    .map(person => person.person_id);
  const removedRigs = (rule.rigs || []).filter(rig => !rigs.has(rig.rig_id))
    .map(rig => rig.rig_id);
  if (rule.people) rule.people = rule.people.filter(person => people.has(person.person_id));
  if (rule.rigs) rule.rigs = rule.rigs.filter(rig => rigs.has(rig.rig_id));
  return {removedPeople, removedRigs};
}
function removedTargetsMessage({removedPeople, removedRigs}) {
  const names = [];
  if (removedPeople.length) names.push(`人物 ${removedPeople.join('、')}`);
  if (removedRigs.length) names.push(`相机组 ${removedRigs.join('、')}`);
  return names.length ? `已移除旧随机规则中不属于当前场景的 ${names.join(' 和 ')} 设置。` : '';
}
async function setScene(scene, saved = false) {
  const removed = reconcileTargets(template(), scene);
  template().base_scene = scene;
  state.scene = scene;
  state.frame = 0;
  state.sceneSaved = saved ? JSON.stringify(scene) : null;
  state.inspected = null;
  map.setReadonly(false);
  map.setMap(await api(`/api/map?environment=${encodeURIComponent(scene.environment)}`), scene.environment);
  sceneEdited(); renderScene(); renderRules();
  return removed;
}
async function setRules(rules) {
  await api('/api/try-rules', {rules});
  state.rules = rules; state.active = 0; state.rulesSaved = JSON.stringify(rules);
  await setScene(state.rules.templates[0].base_scene, false);
  state.rulesSaved = JSON.stringify(rules);
  updateBadge();
}
async function refreshSavedLists() {
  const [recipes, rules] = await Promise.all([api('/api/recipes'), api('/api/rules')]);
  $('recipeSelect').replaceChildren(new Option('选择修订', ''), ...recipes.names.map(name => new Option(name, name)));
  $('ruleSelect').replaceChildren(new Option('选择规则', ''), ...rules.names.map(name => new Option(name, name)));
}
function setTab(which) {
  $('scenePanel').hidden = which !== 'scene';
  $('rulesPanel').hidden = which !== 'rules';
  $('sceneTab').classList.toggle('active', which === 'scene');
  $('rulesTab').classList.toggle('active', which === 'rules');
}
$('sceneTab').onclick = () => setTab('scene');
$('rulesTab').onclick = () => setTab('rules');
$('environment').onchange = async event => {
  try {
    state.scene.environment = event.target.value;
    map.setMap(await api(`/api/map?environment=${encodeURIComponent(state.scene.environment)}`), state.scene.environment);
    sceneEdited(); renderScene();
    show('环境已切换。原人物路线和相机位置仍在旧坐标；如果地图上看不到，请点“把人物和相机整体移到当前环境附近”。');
  } catch (error) { fail(error); }
};
$('placeEnvironment').onclick = async () => {
  try {
    const example = await api(`/api/example?environment=${encodeURIComponent(state.scene.environment)}`);
    const start = state.scene.people[0].route_xy_m[0];
    const target = example.people[0].route_xy_m[0];
    const dx = target[0] - start[0], dy = target[1] - start[1];
    for (const person of state.scene.people) {
      person.route_xy_m = person.route_xy_m.map(([x, y]) => [x + dx, y + dy]);
    }
    for (const rig of state.scene.rigs) {
      rig.origin_m[0] += dx; rig.origin_m[1] += dy;
    }
    sceneEdited(); renderScene(); renderRules();
    show('人物路线和相机组已整体平移到当前环境附近；形状、相对位置和人物组件保留。请检查道路和机位。');
  } catch (error) { fail(error); }
};
for (const [id, field] of [['sceneId', 'scene_id'], ['sceneSeed', 'seed'], ['fps', 'fps'], ['frameCount', 'frame_count']]) {
  $(id).onchange = event => {
    state.scene[field] = id === 'sceneId' ? event.target.value : Number(event.target.value);
    sceneEdited(); renderScene();
  };
}
$('frame').oninput = event => {
  state.frame = Number(event.target.value);
  $('frameLabel').textContent = `${state.frame} · ${(state.frame / state.scene.fps).toFixed(2)} 秒`;
  map.setFrame(state.frame);
  clearPreview('已切换帧；旧预览不代表当前帧。');
};
$('addPerson').onclick = async () => {
  try {
    const current = state.scene.people;
    const example = await api(`/api/example?environment=${encodeURIComponent(state.scene.environment)}`);
    const person = clone(current[0]);
    person.person_id = Math.max(...current.map(p => p.person_id)) + 1;
    const shift = Math.min(current.length, 2) * 0.8;
    person.route_xy_m = example.people[0].route_xy_m.map(([x, y]) => [x + shift, y]);
    person.start_progress_m = example.people[0].start_progress_m;
    current.push(person); sceneEdited(); renderScene(); renderRules();
    show(`人物 ${person.person_id} 已放在当前环境的默认路线附近；身体、动作、衣物和发型沿用原人物，请检查组合与路线。`);
  } catch (error) { fail(error); }
};
$('addRig').onclick = () => {
  let index = 1;
  while (state.scene.rigs.some(r => r.rig_id === `rig_custom_${index}`)) index += 1;
  const rig = clone(state.scene.rigs[0]); rig.rig_id = `rig_custom_${index}`; rig.origin_m[0] += index;
  state.scene.rigs.push(rig); sceneEdited(); renderScene(); renderRules();
};
$('templateSelect').onchange = async event => {
  state.active = Number(event.target.value);
  state.scene = template().base_scene; state.sceneSaved = null; state.frame = 0;
  state.inspected = null; map.setReadonly(false);
  map.setMap(await api(`/api/map?environment=${encodeURIComponent(state.scene.environment)}`), state.scene.environment);
  clearPreview(); renderScene(); renderRules();
  refreshTrajectory();
};
$('newRules').onclick = async () => {
  const saved = state.sceneSaved === JSON.stringify(state.scene);
  const scene = clone(state.scene);
  state.rules = {schema: 'human_batch_v4', seed: scene.seed,
                 templates: [{template_id: `${scene.environment}_baseline`, base_scene: scene}]};
  state.active = 0;
  state.rulesSaved = null;
  $('ruleName').value = `${scene.environment}_variants`;
  try {
    await setScene(scene, saved);
    show(`已从当前${scene.environment === 'riverfront' ? '河岸' : '城市'}场景开始新规则；旧城市规则文件未改。现在设置随机变化，再试抽并导出。`);
  } catch (error) {fail(error);}
};
$('addTemplate').onclick = async () => {
  const names = new Set(state.rules.templates.map(t => t.template_id));
  let name = `${state.scene.scene_id}_copy`, index = 2;
  while (names.has(name)) name = `${state.scene.scene_id}_copy${index++}`;
  const scene = clone(state.scene); scene.scene_id = name;
  state.rules.templates.push({template_id: name, base_scene: scene});
  state.active = state.rules.templates.length - 1;
  await setScene(scene, false); rulesEdited();
};
$('removeTemplate').onclick = async () => {
  if (state.rules.templates.length === 1) return show('至少保留一个基准场景。', true);
  state.rules.templates.splice(state.active, 1); state.active = 0;
  await setScene(template().base_scene, false); rulesEdited();
};
$('templateId').onchange = event => {template().template_id = event.target.value; rulesEdited(); renderRules();};
$('batchSeed').onchange = event => {state.rules.seed = Number(event.target.value); rulesEdited();};
$('check').onclick = async () => {
  try {const result = await api('/api/check', {scene: state.scene}); show(result.status === 'passed' ? '场景 CPU 检查通过。' : result.errors.join('\n'), result.status !== 'passed');}
  catch (error) {fail(error);}
};
$('save').onclick = async () => {
  try {const result = await api('/api/save', {scene: state.scene}); state.sceneSaved = JSON.stringify(state.scene); updateBadge(); await refreshSavedLists(); $('recipeSelect').value = result.id; show(`场景已保存：${result.path}`);}
  catch (error) {fail(error);}
};
$('load').onclick = async () => {
  try {if (!$('recipeSelect').value) throw Error('请先选择一个修订。'); const removed = await setScene(await api(`/api/recipe?id=${encodeURIComponent($('recipeSelect').value)}`), true); show(`已载入场景修订。${removedTargetsMessage(removed)}`);}
  catch (error) {fail(error);}
};
$('sceneFile').onchange = async event => {
  try {
    const scene = JSON.parse(await event.target.files[0].text());
    const checked = await api('/api/check', {scene});
    if (checked.status !== 'passed') throw Error(checked.errors.join('\n'));
    const removed = await setScene(scene, false); show(`已导入场景；尚未保存修订。${removedTargetsMessage(removed)}`);
  } catch (error) {fail(error);}
};
$('tryRules').onclick = async () => {
  try {
    const result = await api('/api/try-rules', {rules: state.rules});
    $('samples').replaceChildren();
    for (const scene of result.scenes) {
      $('samples').append(button(`地图查看 ${scene.scene_id} · ${scene.people.map(p => `P${p.person_id}:${p.body}/${p.motion}`).join('；')}`, () => {
        state.inspected = scene; map.setReadonly(true); refreshTrajectory(scene);
        $('backToBase').hidden = false; $('mapHint').textContent = '正在查看试抽结果；橙色圆点显示当前帧路线位置';
      }));
    }
    show(`已试抽并通过 CPU 检查：${result.scenes.length} 个配置。没有准备几何或渲染。`);
  } catch (error) {fail(error);}
};
$('backToBase').onclick = () => {
  state.inspected = null; map.setReadonly(false);
  trajectoryRequest += 1; clearTimeout(trajectoryTimer);
  if (baseTrajectory) {
    map.setTrajectory(baseTrajectory);
    $('timelineNote').textContent = '地图上的橙色数字圆点显示当前帧的人物位置。';
  } else refreshTrajectory();
  $('backToBase').hidden = true; $('mapHint').textContent = mapHint;
};
$('exportRules').onclick = async () => {
  try {
    const result = await api('/api/export-rules', {name: $('ruleName').value, rules: state.rules});
    state.rulesSaved = JSON.stringify(state.rules); updateBadge(); await refreshSavedLists();
    $('ruleSelect').value = $('ruleName').value;
    const area = $('exportResult'); area.replaceChildren();
    area.append(node('p', `服务器文件：${result.path}`));
    const link = node('a', '下载规则 JSON'); link.href = result.download; area.append(link);
    area.append(node('p', '在项目目录运行：'), node('code', result.command));
    show('规则已导出，前两个变体通过 CPU 检查。批量命令中的 --count 决定每个基准生成多少个；未启动渲染。');
  } catch (error) {fail(error);}
};
$('loadRules').onclick = async () => {
  try {if (!$('ruleSelect').value) throw Error('请先选择规则。'); await setRules(await api(`/api/rule?name=${encodeURIComponent($('ruleSelect').value)}`)); show('已载入规则及其中所有基准场景。');}
  catch (error) {fail(error);}
};
$('ruleFile').onchange = async event => {
  try {await setRules(JSON.parse(await event.target.files[0].text())); show('已导入规则及其中所有基准场景。');}
  catch (error) {fail(error);}
};
async function start(kind) {
  try {
    const revision = state.editRevision;
    const result = await api('/api/start', {kind, scene: state.scene, frame: state.frame, rig_id: $('previewRig').value});
    show(`任务 ${result.task_id} 已启动。`);
    poll(result.task_id, revision);
  } catch (error) {fail(error);}
}
async function poll(id, revision) {
  try {
    const task = await api(`/api/task?id=${encodeURIComponent(id)}`);
    show(`${id} · ${task.state}${task.error ? `\n${task.error}` : ''}`, task.state === 'failed');
    if (task.state === 'completed' && task.kind === 'preview') {
      if (revision !== state.editRevision) return clearPreview();
      for (const camera of ['cam2', 'cam4']) {
        const url = `/api/image/${id}/${camera}?t=${Date.now()}`;
        $(camera).src = url; $(camera + 'Link').href = url;
      }
      $('previewNote').textContent = '点击图片可查看原图。';
    }
    if (!['completed', 'failed'].includes(task.state)) setTimeout(() => poll(id, revision), 1500);
  } catch (error) {fail(error);}
}
$('preview').onclick = () => start('preview');
$('generate').onclick = () => start('run');
window.addEventListener('beforeunload', event => {
  if (state.sceneSaved !== JSON.stringify(state.scene) || state.rulesSaved !== JSON.stringify(state.rules)) event.preventDefault();
});
async function main() {
  const [scene, components] = await Promise.all([api('/api/example?environment=city'), api('/api/components')]);
  state.components = components;
  state.rules = {schema: 'human_batch_v4', seed: scene.seed,
                 templates: [{template_id: scene.scene_id, base_scene: scene}]};
  state.scene = scene;
  await map.setMap(await api('/api/map?environment=city'), 'city');
  await refreshSavedLists(); renderScene(); renderRules();
  refreshTrajectory();
}
main().catch(fail);
